import importlib.util
from pathlib import Path


def test_toy_example_trains_and_keeps_base_frozen():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'toy_train.py'
    spec = importlib.util.spec_from_file_location('toy_train', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main(steps=60)
    assert result['frozen_unchanged'] and result['correct_after'] > result['correct_before']
