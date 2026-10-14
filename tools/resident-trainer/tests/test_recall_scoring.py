from resident_trainer.recall_scoring import score, sibling_leaks

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


def test_toy_example_trains_and_keeps_base_frozen():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples' / 'toy_train.py'
    spec = importlib.util.spec_from_file_location('toy_train', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main(steps=60)
    assert result['frozen_unchanged'] and result['correct_after'] > result['correct_before']
