import os

import pytest

from recovery_lock import RecoveryBusy, recover, recovery_owner


def test_second_owner_is_refused_with_the_first_owners_purpose(tmp_path):
    lock = tmp_path / 'pair.lock'
    with recovery_owner('first', lock):
        with pytest.raises(RecoveryBusy, match='first'):
            with recovery_owner('second', lock):
                pass
    with recovery_owner('third', lock):
        pass  # released after the first owner exits


def test_rerun_is_a_noop_when_a_live_guard_owns_a_healthy_pair(tmp_path):
    lock, owner, restored = tmp_path / 'pair.lock', tmp_path / 'owner.json', []
    owner.write_text('{"pid": %d}' % os.getpid())
    assert recover('x', owner, lambda: True, lambda: restored.append(1), lock) == 'noop'
    assert recover('x', owner, lambda: False, lambda: restored.append(1), lock) == 'restored'
    owner.write_text('{"pid": 999999999}')
    assert recover('x', owner, lambda: True, lambda: restored.append(1), lock) == 'restored'
    assert restored == [1, 1]
