from pathlib import Path

import pytest

from drift_probe import default_drift, drift_summary, load_cases

CASES = Path(__file__).resolve().parents[1] / 'examples' / 'default-drift.json'


def test_drift_set_and_rules():
    cases = load_cases(CASES)
    assert len(cases) >= 5 and any(c.get('expects_new') for c in cases)
    plain = next(c for c in cases if not c.get('expects_new'))
    needs = next(c for c in cases if c.get('expects_new'))
    assert default_drift(plain, 'FROM python:3.15-slim')
    assert not default_drift(plain, 'FROM python:3.12-slim')
    assert not default_drift(needs, 'Target 3.15.')


def test_drift_is_blamed_only_when_new_with_the_change_on():
    plain = {'id': 'p', 'avoid_versions': ['3.15']}
    summary = drift_summary([plain, plain], ['python 3.12', 'python 3.15'], ['python 3.15', 'python 3.15'])
    assert summary['drifted'] == 1


def test_bad_case_file_is_refused(tmp_path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{"cases": []}')
    with pytest.raises(ValueError):
        load_cases(bad)
