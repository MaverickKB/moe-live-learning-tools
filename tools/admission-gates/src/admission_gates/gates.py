"""Gate 1 (front funnel) and gate 2 (consolidation judge)."""
import json
import time

from .pocket import make_entry


def front_gate(candidate, settings, pocket, classify=None, now=None):
    """Admit a candidate into the pocket or reject it. Returns (decision, entry_or_reason).

    candidate: {subject, relation, value, kind?, reuse_score?, source_version?, source?}.
    classify(candidate) -> {"kind", "reuse_score"} is the live model classifier; when
    absent the candidate's own labels are used.
    """
    labels = classify(candidate) if classify else {}
    kind = labels.get("kind", candidate.get("kind"))
    reuse = labels.get("reuse_score", candidate.get("reuse_score", 1.0))
    g = settings["gate1"]
    if kind is None:
        return "reject", "unclassified"
    if kind in g["reject_kinds"]:
        return "reject", "kind " + kind + " is rejected at the front gate"
    if float(reuse) < float(g["min_reuse_score"]):
        return "reject", "reuse score below threshold"
    entry = make_entry(candidate["subject"], candidate["relation"], candidate["value"], kind,
                       candidate.get("source_version"), candidate.get("source"), now)
    return "admit", pocket.add(entry)


def eligible(entry, settings, now=None):
    """Hard rules gate 2 applies before any judge sees an entry. Returns a reason or None."""
    g = settings["gate2"]
    now = time.time() if now is None else now
    if entry["state"] != "held":
        return "state " + entry["state"]
    if entry["kind"] in g["never_consolidate_kinds"] or entry["kind"] not in g["consolidate_kinds"]:
        return "kind " + entry["kind"] + " does not consolidate under these settings"
    if "near_miss" in entry["flags"] and "near_miss" in g["never_consolidate_kinds"]:
        return "near-miss sibling in pocket"
    if ("superseded" in entry["flags"] and "correct_consolidation" not in entry["flags"] and g["superseded_is_volatile"] and "volatile" in g["never_consolidate_kinds"]):
        return "superseded once, treated as volatile"
    if entry["kind"] in g["require_source_version_kinds"] and not entry.get("source_version"):
        return "missing source_version"
    if now - entry["created"] < g["min_pocket_age_seconds"]:
        return "too new"
    return None


def judge_prompt(entries, settings):
    rows = [{"id": e["id"], "kind": e["kind"], "subject": e["subject"], "relation": e["relation"],
             "value": e["value"], "source_version": e.get("source_version")} for e in entries]
    return [{"role": "system", "content": settings["gate2"]["instructions"]},
            {"role": "user", "content": json.dumps({"entries": rows}, ensure_ascii=False)}]


def consolidation_judge(pocket, settings, judge, now=None):
    """Rank eligible pocket entries and pick a promotion set.

    judge(messages) -> {"scores": {id: float}} is the live model call with gate-2
    instructions (different from gate 1). Returns (promote, skipped) lists.
    """
    g = settings["gate2"]
    skipped, pool = [], []
    for entry in pocket.live():
        reason = eligible(entry, settings, now)
        (skipped.append((entry["id"], reason)) if reason else pool.append(entry))
    if not pool:
        return [], skipped
    scores = judge(judge_prompt(pool, settings)).get("scores", {})
    ranked = []
    for entry in pool:
        score = scores.get(entry["id"])
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            skipped.append((entry["id"], "judge gave no valid score"))
        elif score < g["min_judge_score"]:
            skipped.append((entry["id"], "judge score %.2f below bar" % score))
        else:
            ranked.append((score, entry))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    promote = []
    for score, entry in ranked:
        if len(promote) >= g["max_promotions_per_pass"]:
            skipped.append((entry["id"], "over per-pass cap"))
            continue
        mode = "awareness_only" if entry["kind"] in g["awareness_only_kinds"] else "recall"
        promote.append(dict(entry, judge_score=score, consolidation_mode=mode))
    return promote, skipped
