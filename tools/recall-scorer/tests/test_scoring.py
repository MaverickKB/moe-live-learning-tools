from recall_scorer import drop_pct, invented_specifics, score, sibling_leaks
from recall_scorer.scoring import value_hit

VALUES = ['atlas', 'beacon', 'host-a.example', 'blue']


def record(i, prompt, value, text, passed):
    return {'id': i, 'prompt': prompt, 'value': value, 'text': text, 'passed': passed}


def test_strict_needs_grader_pass_and_value():
    rows = [record('a', 'Project codename?', 'atlas', 'It is Atlas.', True),
            record('b', 'Build host?', 'host-a.example', 'Probably a server.', True),
            record('c', 'Favourite colour?', 'blue', "I don't know.", False)]
    result = score(rows, VALUES)
    assert result['strict'] == 1 and result['abstained'] == 1 and result['leaked'] == []


def test_sibling_leak_is_reported_and_tokens_are_whole():
    near_miss = record('d', 'Codename of the second project?', 'beacon', 'It is atlas.', False)
    assert sibling_leaks(near_miss, VALUES) == ['atlas']
    assert sibling_leaks(record('e', 'q', 'x', 'atlases and host-a.example2', False), VALUES) == []
    assert score([near_miss], VALUES)['leaked'] == [('d', ['atlas'])]


def test_invented_specifics_flags_only_unsupported_identifiers():
    facts = 'Library L 2.3 renamed foo() to bar(). The build host is host-a.example.'
    assert invented_specifics('Use bar() from L 2.3 on host-a.example.', facts) == []
    found = invented_specifics('Use L 2.4 on host-c.example at 203.0.113.9 with --fast-mode.', facts)
    assert found == ['--fast-mode', '2.4', '203.0.113.9', 'host-c.example']
    assert invented_specifics('It has 3 parts.', facts) == []


def test_drop_pct():
    assert drop_pct([], lambda a, b: True) is None
    assert drop_pct([(1, 1), (1, 0), (1, 1), (1, 1)], lambda off, on: on < off) == 25.0


def test_empty_reply_scores_as_a_miss():
    # A server can return a reply with no content (text None); it must count as a miss, not crash.
    empty = record('f', 'Project codename?', 'atlas', None, True)
    assert value_hit(empty) is False
    assert sibling_leaks(empty, VALUES) == []
    assert score([empty], VALUES) == {'total': 1, 'strict': 0, 'leaked': [], 'abstained': 0}
