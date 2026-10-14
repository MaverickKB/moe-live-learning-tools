"""Trainer-only admission and paired aborts, without serving lifecycle actions."""

import threading
import time
from pathlib import Path

import torch

from .thin import gather_values

FLOOR_BYTES = 1_000_000_000
VALID_FLAGS = 31  # bits 1|2|4|8|16; any larger word is a misread, never a flag
WORKSPACE_BYTES = 2 * 1024**3


class PassAborted(RuntimeError):
    """The pending trainable update is discarded; the serving owner stays live."""


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('MemAvailable was absent from the owning host')


class ThinPassBudget:
    """Sample local headroom continuously; synchronize aborts at TP boundaries.

    The serving scheduler calls check on both ranks in the same order. Its
    private CUDA workspace includes reserved allocator growth above the
    lease's initial serving allocation. Serving growth makes this estimate
    conservative. No background thread touches the pair communicator.
    """

    def __init__(self, weights, authorized, deadline, cuda_bytes,
                 read_available=available_bytes, floor=FLOOR_BYTES,
                 workspace=WORKSPACE_BYTES, exchange=None):
        self.weights = weights
        self.authorized = authorized
        self.deadline = deadline
        self.cuda_bytes = cuda_bytes
        self.read_available = read_available
        self.floor = floor
        self.workspace = workspace
        # exchange(name, values) -> per-rank values over the trainer-only channel
        self.exchange = exchange
        self.checks = 0
        self.minimum_available = None
        self.peak_allocated_growth = 0
        self.peak_reserved_growth = 0
        self.flags = 0
        self.stop = threading.Event()
        self.monitor = None
        self.lock = threading.Lock()

    def sample(self):
        available = self.read_available()
        allocated, reserved = self.cuda_bytes()
        with self.lock:
            if self.minimum_available is None or available < self.minimum_available:
                self.minimum_available = available
            self.peak_allocated_growth = max(self.peak_allocated_growth,
                                             allocated - self.initial_allocated)
            self.peak_reserved_growth = max(self.peak_reserved_growth,
                                            reserved - self.initial_reserved)
            if available < self.floor:
                self.flags |= 1
            if max(self.peak_allocated_growth, self.peak_reserved_growth) > self.workspace:
                self.flags |= 2
            if time.time() >= self.deadline:
                self.flags |= 4
            if not self.authorized():
                self.flags |= 8

    def watch(self):
        while not self.stop.wait(.05):
            try:
                self.sample()
            except Exception:
                with self.lock:
                    self.flags |= 16

    def check(self):
        try:
            self.sample()
        except Exception:
            with self.lock:
                self.flags |= 16
        with self.lock:
            local = [self.flags, self.minimum_available or 0,
                     self.peak_allocated_growth, self.peak_reserved_growth]
        self.checks += 1
        if self.exchange is not None:
            parts = self.exchange('check-' + str(self.checks), local)
        else:
            packet = torch.tensor(local, device=self.weights.device, dtype=torch.int64)
            parts = gather_values(packet, self.weights).tolist()
        for rank, values in enumerate(parts):
            if len(values) != 4 or not 0 <= int(values[0]) <= VALID_FLAGS:
                raise PassAborted('Invalid trainer flag word from rank ' + str(rank) + ': '
                                  + str(values) + ' (valid flags are 0..31); trainer desync')
        for rank, values in enumerate(parts):
            if values[0]:
                raise PassAborted('Trainer abort on rank ' + str(rank) + ': flags=' + str(values[0]))

    def __enter__(self):
        # Authority is checked before any CUDA callback or allocation.
        if not self.authorized():
            raise PassAborted('Training gate is held')
        self.initial_allocated, self.initial_reserved = 0, 0
        try:
            self.initial_allocated, self.initial_reserved = self.cuda_bytes()
            available = self.read_available()
            # The 17:54 policy retains a 1 GB free-memory floor beside this ceiling.
            if available < self.floor + self.workspace:
                self.flags |= 1
        except Exception:
            self.flags |= 16
        self.check()
        self.monitor = threading.Thread(target=self.watch, name='thin-pass-headroom', daemon=True)
        self.monitor.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        if self.monitor is not None:
            self.monitor.join(timeout=1)
        return False

    def receipt(self):
        with self.lock:
            return {'rank': self.weights.rank, 'minimum_available_bytes': self.minimum_available,
                    'peak_allocated_growth_bytes': self.peak_allocated_growth,
                    'peak_reserved_growth_bytes': self.peak_reserved_growth,
                    'private_workspace_limit_bytes': self.workspace,
                    'protected_floor_bytes': self.floor, 'abort_flags': self.flags,
                    'sampling_interval_seconds': .05}
