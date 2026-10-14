import pytest

from admission_gates.gates import dream_judge, front_gate
from admission_gates.pocket import Pocket
from admission_gates.promotion import run_promotion
from admission_gates.settings import load_settings

CASES = [
    ("a household", "had a one-off event on", "a weekday", "one_off_event", None),
    ("host-a.example", "role is", "build server", "hostname", None),
    ("device A", "room location is", "location 1", "location", None),
    ("person A", "name is", "a person's name", "person_name", None),
    ("the repo", "test command before commit is", "make test", "procedure", None),
    ("library L", "renamed foo() to", "bar()", "api", "libL==2.3"),
    ("library M", "deprecated", "baz()", "api", None),
    ("Python", "latest release is", "3.15", "background", "python==3.15"),
]


def fill(settings):
    pocket = Pocket()
    decisions = {}
    for subject, relation, value, kind, version in CASES:
        decision, detail = front_gate({"subject": subject, "relation": relation, "value": value, "kind": kind,
                                       "source_version": version}, settings, pocket, now=0)
        decisions[kind if kind != "api" else kind + str(version)] = decision
    return pocket, decisions


def judge_all(score):
    def judge(messages):
        assert "dream judge" in messages[0]["content"]
        import json
        return {"scores": {row["id"]: score for row in json.loads(messages[1]["content"])["entries"]}}
    return judge


def test_front_gate_strict_defaults():
    _, decisions = fill(load_settings())
    assert decisions["one_off_event"] == "reject"
    assert all(v == "admit" for k, v in decisions.items() if k != "one_off_event")


def test_judge_promotes_only_stable_general_kinds():
    settings = load_settings()
    pocket, _ = fill(settings)
    promote, skipped = dream_judge(pocket, settings, judge_all(0.9), now=10)
    kinds = sorted((e["kind"], e.get("source_version")) for e in promote)
    assert kinds == [("api", "libL==2.3"), ("background", "python==3.15"), ("procedure", None)]
    assert {e["dream_mode"] for e in promote if e["kind"] == "background"} == {"awareness_only"}
    reasons = dict(skipped).values()
    assert any("missing source_version" in r for r in reasons)
    assert dream_judge(pocket, settings, judge_all(0.5), now=10)[0] == []


def test_near_miss_siblings_stay_in_pocket_even_if_kind_dreams():
    settings = load_settings()
    pocket = Pocket()
    for subject, value in (("device A model", "Model X"), ("device B model", "Model Y")):
        front_gate({"subject": subject, "relation": "is", "value": value, "kind": "procedure"}, settings, pocket, now=0)
    assert all("near_miss" in e["flags"] for e in pocket.live())
    assert dream_judge(pocket, settings, judge_all(1.0), now=1)[0] == []


def test_profile_loosens_capture_but_not_guardrails():
    companion = load_settings({"schema_version": 1, "gate1": {"reject_kinds": ["chatter"]},
                               "gate2": {"dream_kinds": ["codebase", "procedure", "api", "background", "preference"]}})
    pocket = Pocket()
    assert front_gate({"subject": "a household", "relation": "had", "value": "a one-off event",
                       "kind": "one_off_event"}, companion, pocket)[0] == "admit"
    assert front_gate({"subject": "user", "relation": "prefers", "value": "short answers",
                       "kind": "preference"}, companion, pocket, now=0)[0] == "admit"
    promote, _ = dream_judge(pocket, companion, judge_all(0.95), now=1)
    assert [e["kind"] for e in promote] == ["preference"]


def stubs(passing, drops=None):
    calls = {"rollback": 0}

    def dream(entries):
        return "ckpt-1"

    def recall(checkpoint, entries):
        return [{"entry_id": e["id"], "pocket": "off", "passed": e["id"] in passing} for e in entries]

    def regress(checkpoint):
        return drops or {"control_pct": 0.1, "general_chat_pct": 0.2}

    def rollback(checkpoint):
        calls["rollback"] += 1
    return dream, recall, regress, rollback, calls


def test_promotion_frees_only_recalled_entries():
    settings = load_settings()
    pocket, _ = fill(settings)
    promote, _ = dream_judge(pocket, settings, judge_all(0.9), now=10)
    first = promote[0]["id"]
    dream, recall, regress, rollback, calls = stubs({first})
    report = run_promotion(pocket, settings, promote, dream, recall, regress, rollback)
    assert report["freed"] == [first] and len(report["kept"]) == 2
    assert pocket.entries[first]["state"] == "freed" and calls["rollback"] == 0
    assert report["bundle"]["schema_version"] == 1 and report["bundle"]["entries"][0]["id"] == first
    assert all(pocket.entries[i]["state"] == "held" for i in report["kept"])


def test_regression_rolls_back_and_frees_nothing():
    settings = load_settings()
    pocket, _ = fill(settings)
    promote, _ = dream_judge(pocket, settings, judge_all(0.9), now=10)
    ids = {e["id"] for e in promote}
    dream, recall, regress, rollback, calls = stubs(ids, {"control_pct": 3.0, "general_chat_pct": 0.0})
    report = run_promotion(pocket, settings, promote, dream, recall, regress, rollback)
    assert report["rolled_back"] and report["freed"] == [] and calls["rollback"] == 1
    assert all(pocket.entries[i]["state"] == "held" for i in ids)


def test_free_requires_pocket_off_receipt():
    settings = load_settings()
    pocket, _ = fill(settings)
    entry = pocket.live()[0]
    with pytest.raises(PermissionError):
        pocket.free(entry["id"], {"entry_id": entry["id"], "pocket": "on", "passed": True})


def test_supersede_in_pocket_and_correct_a_dreamed_fact():
    settings = load_settings()
    pocket = Pocket()
    fact = {"subject": "library L", "relation": "default port is", "value": "8000", "kind": "api",
            "source_version": "libL==2.3"}
    _, old = front_gate(fact, settings, pocket, now=0)
    _, new = front_gate(dict(fact, value="8080"), settings, pocket, now=0)
    assert new["value"] == "8080" and "superseded" in new["flags"] and "correct_dream" not in new["flags"]
    assert [e["state"] for e in pocket.entries.values() if e["superseded_by"] == new["id"]] == ["superseded"]
    assert dream_judge(pocket, settings, judge_all(1.0), now=1)[0] == []  # volatile under defaults
    # A dreamed-and-freed fact corrected later must dream again over the old value.
    pocket2 = Pocket()
    _, entry = front_gate(fact, settings, pocket2, now=0)
    pocket2.free(entry["id"], {"entry_id": entry["id"], "pocket": "off", "passed": True})
    _, fix = front_gate(dict(fact, value="8080"), settings, pocket2, now=0)
    assert "correct_dream" in fix["flags"] and fix["corrects_value"] == "8000"
    promote, _ = dream_judge(pocket2, settings, judge_all(1.0), now=1)
    assert [e["id"] for e in promote] == [fix["id"]]
    # A different source_version is a separate fact, not a supersede.
    _, other = front_gate(dict(fact, value="9000", source_version="libL==3.0"), settings, pocket2, now=0)
    assert other["id"] != fix["id"] and "superseded" not in other["flags"]
