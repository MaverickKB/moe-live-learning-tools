"""Tier 2: consolidate one promotion set, verify pocket-off, free or roll back.

All heavy steps are injected so this runs on CPU with stubs:
  consolidate(entries) -> checkpoint id      (trained expert path or write-not-train path)
  recall(checkpoint, entries) -> [{"entry_id", "pocket": "off", "passed"}]
      For a correction entry, passed means the new value is recalled and the
      old value (entry["corrects_value"]) is not. Awareness-only entries also
      need a default-drift probe; that probe is live-only (untested).
  regress(checkpoint) -> {"control_pct": float, "general_chat_pct": float}  (drop, %)
  rollback(checkpoint) -> None
"""
from .versioning import stamp


def run_promotion(pocket, settings, promote, consolidate, recall, regress, rollback):
    g = settings["guardrails"]
    report = {"promoted": [e["id"] for e in promote], "freed": [], "kept": [], "rolled_back": False}
    if not promote:
        return report
    ids = [e["id"] for e in promote]
    pocket.mark(ids, "consolidating")
    for i in ids:
        pocket.entries[i]["consolidation_attempts"] += 1
    checkpoint = consolidate(promote)
    report["checkpoint"] = checkpoint
    drops = regress(checkpoint)
    report["regression"] = drops
    if (drops.get("control_pct") is None or drops.get("general_chat_pct") is None
            or drops["control_pct"] > g["max_control_regression_pct"]
            or drops["general_chat_pct"] > g["max_general_regression_pct"]):
        rollback(checkpoint)
        pocket.mark(ids, "held")
        report.update(rolled_back=True, kept=ids)
        return report
    receipts = {r["entry_id"]: r for r in recall(checkpoint, promote)}
    for entry in promote:
        receipt = receipts.get(entry["id"])
        if receipt and receipt.get("pocket") == "off" and receipt.get("passed") is True:
            pocket.free(entry["id"], receipt)
            report["freed"].append(entry["id"])
        else:
            pocket.mark([entry["id"]], "held")
            report["kept"].append(entry["id"])
    report["bundle"] = stamp({"checkpoint": checkpoint,
                              "entries": [{"id": e["id"], "kind": e["kind"], "source_version": e.get("source_version"),
                                           "consolidation_mode": e["consolidation_mode"]} for e in promote if e["id"] in report["freed"]]},
                             "consolidation_bundle")
    return report
