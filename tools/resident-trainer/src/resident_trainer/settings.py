"""Runtime settings, read from environment variables with generic defaults.

Nothing here names a user, host or path on any particular machine.
"""

import os
from pathlib import Path


def receipts_dir():
    """Where per-rank job receipts, live phase logs and stack dumps are written."""
    path = Path(os.environ.get('RESIDENT_TRAINER_RECEIPTS', 'receipts'))
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir():
    """Where per-rank added-neuron checkpoints are written after each update."""
    path = Path(os.environ.get('RESIDENT_TRAINER_CHECKPOINTS', 'checkpoints'))
    path.mkdir(parents=True, exist_ok=True)
    return path


def authority():
    """The exact operator string a job must carry to run. Empty means no job may run."""
    return os.environ.get('RESIDENT_TRAINER_AUTHORITY', '')


# Layer that receives the added shared-expert neurons, and the deeper layer whose
# tensors are also included in the frozen-weight hash. Values from the origin run
# (a 46-layer MoE); change them for another model.
TRAIN_LAYER = int(os.environ.get('RESIDENT_TRAINER_LAYER', '35'))
WATCH_LAYER = int(os.environ.get('RESIDENT_TRAINER_WATCH_LAYER', '44'))
