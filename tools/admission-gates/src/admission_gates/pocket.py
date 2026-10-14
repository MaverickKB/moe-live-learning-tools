"""Tier 1: the persistent catch-all pocket index.

This is the bookkeeping layer over the attention-memory pocket: what is held, why,
its version, what superseded it, and whether a dream has absorbed it. Encoding an
entry into attention-memory slots is done by the existing pocket build path; this
module never frees an entry without a passing pocket-off recall receipt.
"""
import hashlib
import json
import time
from pathlib import Path

from .files import write_json
from .versioning import check, stamp

STATES = ("held", "dreaming", "dreamed", "freed", "superseded")


def entry_id(subject, relation, source_version=None):
    raw = json.dumps([subject.strip().lower(), relation.strip().lower(), source_version], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_entry(subject, relation, value, kind, source_version=None, source=None, now=None):
    for name, field in (("subject", subject), ("relation", relation), ("value", value)):
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Pocket entry needs a nonempty " + name)
    return stamp({"id": entry_id(subject, relation, source_version), "subject": subject, "relation": relation,
                  "value": value, "kind": kind, "source_version": source_version, "source": source,
                  "state": "held", "flags": [], "supersedes": [], "superseded_by": None,
                  "dream_attempts": 0, "created": now if now is not None else time.time()}, "pocket_entry")


def _tokens(text):
    return text.lower().split()


def near_miss(a, b):
    """Sibling facts: same relation, different value, subjects differ by one word."""
    if a["relation"].lower() != b["relation"].lower() or a["value"].lower() == b["value"].lower():
        return False
    ta, tb = _tokens(a["subject"]), _tokens(b["subject"])
    return len(ta) == len(tb) and sum(x != y for x, y in zip(ta, tb)) == 1


class Pocket:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.entries = {}
        if self.path and self.path.exists():
            self.load(json.loads(self.path.read_text(encoding="utf-8")))

    def live(self):
        return [e for e in self.entries.values() if e["state"] in ("held", "dreaming", "dreamed")]

    def add(self, entry):
        """Insert, superseding an older live entry with the same subject, relation and version."""
        check(entry, "pocket_entry")
        old = self.entries.get(entry["id"])
        entry = dict(entry, flags=list(entry["flags"]), supersedes=list(entry["supersedes"]))
        if old and old["state"] in ("held", "dreaming", "dreamed", "freed"):
            if old["value"] == entry["value"]:
                return old
            old_id = old["id"] + "@" + str(len(old["supersedes"]) + 1)
            retired = dict(old, id=old_id, state="superseded", superseded_by=entry["id"])
            self.entries[old_id] = retired
            entry["supersedes"] = old["supersedes"] + [old_id]
            if "superseded" not in entry["flags"]:
                entry["flags"].append("superseded")
            if old["state"] in ("dreamed", "freed"):
                # The old value lives in the weights: the new value must dream over it.
                entry["flags"].append("correct_dream")
                entry["corrects_value"] = old["value"]
        for other in self.live():
            if other["id"] != entry["id"] and near_miss(other, entry):
                for item in (other, entry):
                    if "near_miss" not in item["flags"]:
                        item["flags"].append("near_miss")
        self.entries[entry["id"]] = entry
        self.save()
        return entry

    def mark(self, ids, state):
        if state not in STATES:
            raise ValueError("Unknown pocket state " + state)
        for i in ids:
            self.entries[i]["state"] = state
        self.save()

    def free(self, entry_id_value, receipt):
        """Free an entry only with a passing pocket-off recall receipt for it."""
        entry = self.entries[entry_id_value]
        if (not isinstance(receipt, dict) or receipt.get("entry_id") != entry_id_value
                or receipt.get("pocket") != "off" or receipt.get("passed") is not True):
            raise PermissionError("Freeing needs a passing pocket-off recall receipt for " + entry_id_value)
        entry["state"] = "freed"
        entry["freed_by"] = receipt
        self.save()

    def bundle(self):
        return stamp({"entries": sorted(self.entries.values(), key=lambda e: e["id"])}, "pocket_bundle")

    def load(self, bundle):
        check(bundle, "pocket_bundle")
        for entry in bundle["entries"]:
            check(entry, "pocket_entry")
        self.entries = {e["id"]: e for e in bundle["entries"]}

    def save(self):
        if self.path:
            write_json(self.path, self.bundle())
