"""No-restart code loading and job authority, CPU only (no GPU or serving)."""
import hashlib
import importlib.util
import sys


from resident_trainer.hotload import hotload, local_verdict
from resident_trainer.thin_job import job_authorized

GO = 'OPERATOR GO example-window'


def package(tmp_path, body='VALUE = 2\n'):
    pkg = tmp_path / 'hotpkg'
    pkg.mkdir()
    (pkg / '__init__.py').write_text('')
    (pkg / 'fit.py').write_text('VALUE = 1\n')
    sys.path.insert(0, str(tmp_path))
    module = importlib.import_module('hotpkg.fit')
    (pkg / 'fit.py').write_text(body)
    return pkg, module, {'fit.py': hashlib.sha256(body.encode()).hexdigest()}


import importlib  # noqa: E402


def pair_exchange(peer):
    return lambda name, value: [value, peer(name, value)]


def test_hotload_reloads_in_place_when_both_ranks_agree(tmp_path):
    pkg, module, expected = package(tmp_path)
    try:
        report = hotload('hotpkg', pkg, expected, pair_exchange(lambda n, v: v))
        assert report == {'hotload': 'applied', 'reloaded': ['fit'], 'serving_restart': False}
        assert module.VALUE == 2  # same module object, new code
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.startswith('hotpkg')]:
            del sys.modules[name]


def test_hotload_refuses_on_peer_mismatch_and_restart_only_files(tmp_path):
    pkg, module, expected = package(tmp_path)
    try:
        bad_peer = lambda n, v: {'ok': False, 'problems': ['staged hash differs'], 'hashes': {}}  # noqa: E731
        assert hotload('hotpkg', pkg, expected, pair_exchange(bad_peer))['hotload'] == 'refused'
        assert module.VALUE == 1
        verdict = local_verdict(pkg, {'autograd_native.py': '0' * 64, 'fit.py': '0' * 64})
        assert not verdict['ok'] and any('restart' in p for p in verdict['problems'])
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.startswith('hotpkg')]:
            del sys.modules[name]


def test_jobs_need_the_configured_authority(tmp_path, monkeypatch):
    monkeypatch.delenv('RESIDENT_TRAINER_AUTHORITY', raising=False)
    assert not job_authorized({'mode': 'update', 'authority': GO}, tmp_path)  # unset: nothing runs
    monkeypatch.setenv('RESIDENT_TRAINER_AUTHORITY', GO)
    assert job_authorized({'mode': 'hotload', 'authority': GO}, tmp_path)
    assert job_authorized({'mode': 'update', 'authority': GO}, tmp_path)
    assert not job_authorized({'mode': 'update', 'authority': 'something else'}, tmp_path)
    (tmp_path / 'STOP').write_text('')
    assert not job_authorized({'mode': 'update', 'authority': GO}, tmp_path)
