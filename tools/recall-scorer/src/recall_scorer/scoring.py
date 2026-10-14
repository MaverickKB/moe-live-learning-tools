"""Offline pocket-off scoring of recall answers produced with the retrieval pocket switched off.

The live scorer in the origin project drove a two-rank server; this module keeps
only the engine-free part: given saved answers, count strict hits and detect
sibling-value leaks (an answer that names a value belonging to a different fact).

Each answer record is a dict with `id`, `prompt`, `value` (the expected value)
and `text` (the model's answer). `passed` is set by your own grader; strict
credit requires both `passed` and the expected value appearing in the text.
"""

import re

ABSTAIN = re.compile(r"don't have|do not have|no information|not (?:available|known|specified)"
                     r"|unknown|i don't know", re.I)


def contains(text, value):
    """Whole-token, case-insensitive match (hyphens count as part of a token)."""
    return re.search(r'(?<![\w-])' + re.escape(value.lower()) + r'(?![\w-])', text.lower()) is not None


def value_hit(record):
    return contains(record['text'], record['value'])


def sibling_leaks(record, values):
    """Values from other facts that appear in the answer but not in its prompt or expected value."""
    expected, prompt = record['value'].lower(), record['prompt'].lower()
    return [v for v in values
            if contains(record['text'], v) and v.lower() not in expected and v.lower() not in prompt]


def score(records, values):
    """Summarize one suite: strict hits, failed answers that leaked a sibling, abstentions."""
    strict = sum(bool(r.get('passed')) and value_hit(r) for r in records)
    leaked = [(r['id'], sibling_leaks(r, values)) for r in records
              if not r.get('passed') and sibling_leaks(r, values)]
    abstained = sum(bool(ABSTAIN.search(r['text'])) for r in records)
    return {'total': len(records), 'strict': strict, 'leaked': leaked, 'abstained': abstained}


# Invented specifics: concrete identifiers in an answer that appear in neither the
# reference facts nor the question. A regex heuristic; it misses inventions in plain words.
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


def invented_specifics(answer, *allowed_texts):
    allowed = set()
    for text in allowed_texts:
        allowed |= specifics(text)
    return sorted(s for s in specifics(answer) - allowed if not s.isdigit())


def drop_pct(pairs, worse):
    """pairs: [(off, on)]. worse(off, on) -> bool. Returns percent of pairs that got worse."""
    if not pairs:
        return None
    return 100.0 * sum(bool(worse(off, on)) for off, on in pairs) / len(pairs)
