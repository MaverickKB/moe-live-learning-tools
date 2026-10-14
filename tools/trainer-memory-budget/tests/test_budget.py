"""CPU proofs: admission, paired aborts, and the floor that aborts only the trainer."""

import time
from types import SimpleNamespace

import pytest

from trainer_budget import FLOOR_BYTES, PassAborted, TrainerBudget

WEIGHTS = SimpleNamespace(rank=0)
PLENTY = FLOOR_BYTES + 3 * 1024**3


def mirror(name, local):
    return [local, list(local)]


def test_hold_rejects_before_any_native_work():
    def forbidden():
        pytest.fail('Held work reached a native dependency')

    budget = TrainerBudget(None, lambda: False, time.time() + 60, forbidden,
                           read_available=forbidden, exchange=mirror)
    with pytest.raises(PassAborted, match='held'):
        with budget:
            pytest.fail('Held work entered a lease')


def test_peer_breach_aborts_this_rank_at_the_same_boundary():
    budget = TrainerBudget(WEIGHTS, lambda: True, time.time() + 60, lambda: (10, 10),
                           read_available=lambda: PLENTY,
                           exchange=lambda name, local: [local, [1, 0, 0, 0]])
    with pytest.raises(PassAborted, match='rank 1'):
        with budget:
            pytest.fail('Peer breach entered training')
    assert budget.monitor is None


def test_reserved_growth_and_decimal_floor():
    current, available = [10, 10], [FLOOR_BYTES + 100]
    budget = TrainerBudget(WEIGHTS, lambda: True, time.time() + 60, lambda: tuple(current),
                           read_available=lambda: available[0], workspace=50, exchange=mirror)
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


def test_garbage_flag_words_are_treated_as_desync():
    budget = TrainerBudget(WEIGHTS, lambda: True, time.time() + 60, lambda: (1, 1),
                           read_available=lambda: PLENTY,
                           exchange=lambda name, local: [local, [45309, 0, 0, 0]])
    with pytest.raises(PassAborted, match='Invalid trainer flag word'):
        with budget:
            pass


def test_deadline_and_revoked_authority_abort():
    allowed = [True]
    budget = TrainerBudget(WEIGHTS, lambda: allowed[0], time.time() + 60, lambda: (0, 0),
                           read_available=lambda: PLENTY, exchange=mirror)
    with budget:
        allowed[0] = False
        with pytest.raises(PassAborted, match='flags=8'):
            budget.check()
