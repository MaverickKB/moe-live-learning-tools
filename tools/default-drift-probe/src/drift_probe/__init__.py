"""Default-drift probe: knowing a new version exists must not change default choices.

`default_drift(case, text)` is True when an answer uses a version the case says must
not be the default. `drift_summary` blames drift on the change under test only when it
appears with the change on and not with it off.
"""
import json
import re
from pathlib import Path

SPECIFIC = re.compile(
    r"(?<![\w.])(?:"
    r"\d{1,3}(?:\.\d{1,3}){3}"                    # IPv4
    r"|v?\d+\.\d+(?:\.\d+)*(?:[a-z]+\d*)?"         # versions like 3.12, v2.3.1, 3.15a1
    r"|[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:example|local|lan|internal|com|org|net|io)"  # hostnames
    r"|(?:/[\w.-]+){2,}"                           # paths
    r"|--?[a-z][\w-]{2,}"                          # CLI flags
    r"|[a-z]+-?\d+[a-z0-9]*"                       # model codes like a7, x200
    r")(?![\w])", re.I)


def specifics(text):
    return {m.group(0).lower().rstrip(".") for m in SPECIFIC.finditer(text or "")}


def default_drift(case, text):
    """True when the answer uses a version the case says must not be the default."""
    found = specifics(text)
    if case.get("expects_new"):
        return False  # the prompt needs the new feature; using it is correct
    return any(v.lower() in found for v in case["avoid_versions"])


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        raise ValueError("eval set needs schema_version 1 and cases")
    return data["cases"]


def drift_summary(cases, off_texts, on_texts):
    """Drift caused by the expert: drifted with expert on but not off."""
    rows = []
    for case, off, on in zip(cases, off_texts, on_texts):
        rows.append({"id": case["id"], "off": default_drift(case, off), "on": default_drift(case, on)})
    caused = sum(r["on"] and not r["off"] for r in rows)
    return {"drifted": caused, "total": len(rows), "rows": rows}
