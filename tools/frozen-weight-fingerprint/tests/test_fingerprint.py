from types import SimpleNamespace

import torch

from frozen_fingerprint import fingerprint, frozen_sample


def model():
    torch.manual_seed(0)
    layer = SimpleNamespace(attn={'q': torch.randn(8, 8)}, experts=[torch.randn(4, 8, dtype=torch.bfloat16),
                                                                   torch.randint(0, 255, (16,), dtype=torch.uint8)])
    return SimpleNamespace(norm=torch.ones(8), layers=[layer, SimpleNamespace(w=torch.randn(3))])


def test_any_single_byte_change_alters_the_digest():
    weights = model()
    sample = frozen_sample(weights, layers=(0,))
    before = fingerprint(sample)
    assert fingerprint(sample) == before
    weights.layers[0].experts[1][5] += 1
    assert fingerprint(sample) != before


def test_bf16_low_bit_flip_detected_and_chunks_agree():
    weights = model()
    sample = frozen_sample(weights, layers=(0,))
    before = fingerprint(sample, chunk=7)
    raw = weights.layers[0].experts[0].view(torch.int16).reshape(-1)
    raw[3] ^= 1
    assert fingerprint(sample, chunk=7) != before


def test_sample_walks_nested_containers_and_respects_limit():
    names = [n for n, _ in frozen_sample(model(), layers=(0, 1, 9))]
    assert names[0] == 'norm' and any('experts' in n for n in names) and any(n.startswith('layer1') for n in names)
    assert len(frozen_sample(model(), layers=(0,), limit=2)) == 2
