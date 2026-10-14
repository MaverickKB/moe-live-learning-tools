import json
from pathlib import Path

import pytest

from admission_gates.pocket import Pocket, make_entry
from admission_gates.settings import GuardrailError, load_settings
from admission_gates.versioning import API_VERSION, VersionMismatch, check, stamp

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_strict_defaults_and_example_files_load():
    default = load_settings()
    assert "one_off_event" in default["gate1"]["reject_kinds"]
    assert "person_name" in default["gate2"]["never_dream_kinds"]
    assert load_settings(EXAMPLES / "settings-default.json")["gate2"] == default["gate2"] | {
        k: v for k, v in json.loads((EXAMPLES / "settings-default.json").read_text())["gate2"].items()}
    companion = load_settings(EXAMPLES / "profile-companion.json")
    assert "personal_detail" not in companion["gate1"]["reject_kinds"]
    assert companion["guardrails"] == default["guardrails"]


@pytest.mark.parametrize("override", [
    {"require_pocket_off_recall": False}, {"require_control_regression": False},
    {"require_general_chat_regression": False}, {"max_general_regression_pct": 5.0},
    {"keep_rollback_checkpoints": 0}])
def test_guardrails_cannot_be_loosened(override):
    with pytest.raises(GuardrailError):
        load_settings({"schema_version": 1, "guardrails": override})


def test_guardrails_can_be_tightened_and_unknown_keys_refused():
    assert load_settings({"schema_version": 1, "guardrails": {"max_control_regression_pct": 0.2}})
    with pytest.raises(ValueError):
        load_settings({"schema_version": 1, "gate2": {"surprise": 1}})
    with pytest.raises(ValueError):
        load_settings({"schema_version": 1, "gate2": {"dream_kinds": ["hostname"]}})


def test_versions_are_stamped_and_checked(tmp_path):
    entry = make_entry("library L", "renamed foo() to", "bar()", "api", source_version="libL==2.3")
    assert entry["schema_version"] == 1 and entry["api_version"] == API_VERSION
    assert entry["source_version"] == "libL==2.3"
    pocket = Pocket(tmp_path / "pocket.json")
    pocket.add(entry)
    bundle = json.loads((tmp_path / "pocket.json").read_text())
    assert bundle["schema_version"] == 1 and bundle["api_version"] == API_VERSION
    assert Pocket(tmp_path / "pocket.json").entries.keys() == pocket.entries.keys()
    with pytest.raises(VersionMismatch):
        check(dict(bundle, schema_version=2), "pocket_bundle")
    with pytest.raises(VersionMismatch):
        check(dict(stamp({}, "dream_bundle"), api_version="2.0"), "dream_bundle")
    with pytest.raises(VersionMismatch):
        check({"api_version": API_VERSION}, "pocket_entry")
