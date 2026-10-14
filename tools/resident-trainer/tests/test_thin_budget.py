"""CPU admission and paired aborts leave serving and frozen weights alone."""

import time
from types import SimpleNamespace

import pytest

from resident_trainer.thin_budget import FLOOR_BYTES, PassAborted, ThinPassBudget


def test_hold_rejects_before_cuda_or_collective_work():
    def forbidden():
        pytest.fail('Held work reached a native dependency')

    budget = ThinPassBudget(None, lambda: False, time.time() + 60, forbidden,
                            read_available=forbidden)
    with pytest.raises(PassAborted, match='held'):
        with budget:
            pytest.fail('Held work entered a lease')


def test_remote_breach_aborts_local_pass_at_a_paired_boundary():
    class Peer:
        def all_gather(self, local, output):
            output[:4].copy_(local)
            output[4:].copy_(local)
            output[4] = 1

    weights = SimpleNamespace(world=2, rank=0, device='cpu', comm=Peer())
    budget = ThinPassBudget(weights, lambda: True, time.time() + 60,
                            lambda: (10, 10), read_available=lambda: FLOOR_BYTES + 3 * 1024**3)
    with pytest.raises(PassAborted, match='rank 1'):
        with budget:
            pytest.fail('Peer breach entered training')
    assert budget.monitor is None


def test_budget_includes_reserved_growth_and_preserves_the_decimal_floor():
    class Peer:
        def all_gather(self, local, output):
            output[:4].copy_(local)
            output[4:].copy_(local)

    weights = SimpleNamespace(world=2, rank=0, device='cpu', comm=Peer())
    current = [10, 10]
    available = [FLOOR_BYTES + 100]
    budget = ThinPassBudget(weights, lambda: True, time.time() + 60,
                            lambda: tuple(current), read_available=lambda: available[0], workspace=50)
    with budget:
        current[1] = 61
        with pytest.raises(PassAborted, match='flags=2'):
            budget.check()
        available[0] = FLOOR_BYTES - 1
        with pytest.raises(PassAborted, match='flags=3'):
            budget.check()
    receipt = budget.receipt()
    assert receipt['minimum_available_bytes'] == FLOOR_BYTES - 1
    assert receipt['peak_reserved_growth_bytes'] == 51
    assert not budget.monitor.is_alive()
