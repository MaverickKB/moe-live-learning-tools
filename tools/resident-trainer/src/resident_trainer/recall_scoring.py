"""Offline scoring of recall answers produced with the retrieval pocket switched off.

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
    """Whole-token, case-insensitive match (hyphens count as part of a token).

    A missing answer (None, e.g. an empty reply from the server) contains nothing.
    """
    return re.search(r'(?<![\w-])' + re.escape(value.lower()) + r'(?![\w-])', (text or '').lower()) is not None


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
    abstained = sum(bool(ABSTAIN.search(r['text'] or '')) for r in records)
    return {'total': len(records), 'strict': strict, 'leaked': leaked, 'abstained': abstained}
