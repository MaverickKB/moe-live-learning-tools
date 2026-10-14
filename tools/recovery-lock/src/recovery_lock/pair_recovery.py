"""Single recovery owner for the TP=2 pair.

Two restarts once raced: the retained pair guard's rollback and a
direct container restart both bound the master port (EADDRINUSE). Every recovery path
must hold this one exclusive lock, which covers the master port and
pair-guard-owner.json. Re-running a recovery is safe: when a live guard
already owns the pair and serving is healthy, the recovery is a no-op.
"""

import contextlib
import errno
import fcntl
import json
import os
import time
from pathlib import Path

MASTER_PORT = int(os.environ.get('RECOVERY_MASTER_PORT', '29551'))
LOCK_PATH = Path(os.environ.get('RECOVERY_LOCK_PATH', 'state/locks/pair-recovery.lock'))


class RecoveryBusy(RuntimeError):
    """Another process already owns pair recovery; do not start a second restart."""


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def guard_owner(owner_path):
    """The live guard pid recorded in pair-guard-owner.json, or None."""
    try:
        owner = json.loads(Path(owner_path).read_text())
    except (FileNotFoundError, ValueError):
        return None
    pid = owner.get('pid')
    return pid if pid and pid_alive(pid) else None


@contextlib.contextmanager
def recovery_owner(purpose, lock_path=LOCK_PATH):
    """Hold the exclusive pair-recovery lock; raise RecoveryBusy if held."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, 'a+')
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EAGAIN, errno.EACCES):
                handle.seek(0)
                raise RecoveryBusy('Pair recovery already owned: ' + handle.read().strip()) from error
            raise
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({'pid': os.getpid(), 'purpose': purpose, 'port': MASTER_PORT,
                                 'since': time.time()}))
        handle.flush()
        yield
    finally:
        handle.close()


def recover(purpose, owner_path, healthy, restore, lock_path=LOCK_PATH):
    """Idempotent single-owner recovery.

    healthy() -> bool checks serving; restore() performs the canonical restore.
    Returns 'noop' when a live guard owns a healthy pair, otherwise 'restored'.
    """
    with recovery_owner(purpose, lock_path):
        if guard_owner(owner_path) is not None and healthy():
            return 'noop'
        restore()
        return 'restored'
