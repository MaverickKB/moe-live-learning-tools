"""Small hashing helpers shared by the scope and checkpoint checks."""

import hashlib
import json
from pathlib import Path


def digest(value):
    """Canonical SHA-256 of a JSON-serialisable value."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
