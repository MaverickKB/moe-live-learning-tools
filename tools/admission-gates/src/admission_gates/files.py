"""Atomic JSON writes."""
import json
import os
import tempfile
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix="."+path.name+"-",suffix=".tmp",dir=path.parent)
    temporary = Path(name)
    with os.fdopen(descriptor, "w") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)
