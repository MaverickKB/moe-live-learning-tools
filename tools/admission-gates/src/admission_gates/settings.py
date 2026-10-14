"""Gate settings with strict defaults, profile overrides and locked guardrails."""
import copy
import json
from pathlib import Path

from .versioning import SCHEMAS, check

KINDS = ("one_off_event", "chatter", "location", "hostname", "device", "near_miss",
         "person_name", "volatile", "personal_detail", "preference",
         "codebase", "procedure", "api", "background")

DEFAULTS = {
    "schema_version": SCHEMAS["admission_settings"],
    "profile": "default",
    "gate1": {
        "reject_kinds": ["one_off_event", "chatter", "personal_detail"],
        "min_reuse_score": 0.5,
        "classifier_instructions": (
            "You are the front funnel for a memory pocket. Classify one candidate fact by kind "
            "and estimate how often future tasks would reuse it (0..1). One-off events and chatter "
            "score low. Return only JSON {\"kind\": \"<kind>\", \"reuse_score\": <0..1>}."),
    },
    "gate2": {
        "dream_kinds": ["codebase", "procedure", "api", "background"],
        "never_dream_kinds": ["location", "hostname", "device", "near_miss", "person_name", "volatile"],
        "require_source_version_kinds": ["api"],
        "awareness_only_kinds": ["background"],
        "min_judge_score": 0.8,
        "min_pocket_age_seconds": 0,
        "max_promotions_per_dream": 50,
        "superseded_is_volatile": True,
        "instructions": (
            "You are the dream judge. Rank pocket entries for permanent weight storage. "
            "Prefer stable, general, non-volatile knowledge that many future tasks reuse: "
            "codebases, procedures, versioned APIs, background facts. Score low anything "
            "personal, volatile, location-like, easily confused with a sibling fact, or one-off. "
            "Knowing a new version exists must not change default choices. "
            "Return JSON {\"scores\": {\"<entry id>\": <0..1>}}."),
    },
    "backend": {"kind": "none", "endpoint": None, "model": "your-model", "timeout": 120},
    "guardrails": {
        "require_pocket_off_recall": True,
        "require_control_regression": True,
        "require_general_chat_regression": True,
        "max_control_regression_pct": 1.0,
        "max_general_regression_pct": 1.0,
        "keep_rollback_checkpoints": 3,
    },
}

LOCKED_TRUE = ("require_pocket_off_recall", "require_control_regression", "require_general_chat_regression")
CEILINGS = {"max_control_regression_pct": 1.0, "max_general_regression_pct": 1.0}
FLOORS = {"keep_rollback_checkpoints": 1}


class GuardrailError(ValueError):
    pass


def _merge(base, over, path=""):
    for key, value in over.items():
        if key == "_comments":
            continue
        if path == "" and key not in base:
            raise ValueError("Unknown settings section: " + key)
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value, path + key + ".")
        else:
            if path and key not in base:
                raise ValueError("Unknown setting: " + path + key)
            base[key] = copy.deepcopy(value)


def validate(settings):
    check(settings, "admission_settings")
    g = settings["guardrails"]
    for key in LOCKED_TRUE:
        if g.get(key) is not True:
            raise GuardrailError(key + " cannot be turned off")
    for key, ceiling in CEILINGS.items():
        if not 0 <= float(g[key]) <= ceiling:
            raise GuardrailError("%s must be between 0 and %s" % (key, ceiling))
    for key, floor in FLOORS.items():
        if int(g[key]) < floor:
            raise GuardrailError("%s must be at least %d" % (key, floor))
    for section, key in (("gate1", "reject_kinds"), ("gate2", "dream_kinds"), ("gate2", "never_dream_kinds"),
                         ("gate2", "require_source_version_kinds"), ("gate2", "awareness_only_kinds")):
        unknown = set(settings[section][key]) - set(KINDS)
        if unknown:
            raise ValueError("%s.%s has unknown kinds %s" % (section, key, sorted(unknown)))
    overlap = set(settings["gate2"]["dream_kinds"]) & set(settings["gate2"]["never_dream_kinds"])
    if overlap:
        raise ValueError("Kinds both dream and never dream: %s" % sorted(overlap))
    if settings["backend"]["kind"] not in ("none", "serving"):
        raise ValueError("backend.kind must be none or serving")
    if not 0 <= float(settings["gate2"]["min_judge_score"]) <= 1:
        raise ValueError("min_judge_score must be 0..1")
    return settings


def load_settings(profile=None):
    """Strict defaults, optionally overridden by a profile dict or JSON path."""
    settings = copy.deepcopy(DEFAULTS)
    if profile is not None:
        if not isinstance(profile, dict):
            profile = json.loads(Path(profile).read_text(encoding="utf-8"))
        check(profile, "admission_settings")
        _merge(settings, profile)
    return validate(settings)
