"""CPU source-contract checks of resident ownership and native head tiling."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from resident_trainer.resident_math import kda_reference
from resident_trainer.thin_native import ResidentHead, resident_identity


@dataclass
class F8:
    weight: torch.Tensor
    scale: torch.Tensor
    n: int
    k: int


def test_head_views_existing_storage_and_keeps_original_split_k():
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(inputs, weight, scale, output, partial, rows, stride, **options):
                calls.append({'grid': grid, 'weight': weight, 'scale': scale, **options})
                output.copy_(inputs.float() @ weight.T)
            return launch

    def split_k(rows, hidden):
        return 1 if rows == 19 else 4

    qmm = SimpleNamespace(b16_split_k=split_k, bucket=lambda rows: 16,
                          F8_CONFIG={16: (4, 6)}, F8_BN_DECODE=32, B16_BN=64,
                          B16_BK=64, _seq=lambda *args: False,
                          _prompt_cfg=lambda *args: None, _fmm=Kernel())
    weight = torch.randn(19, 4)
    scale = torch.ones(19, 1)
    head = F8(weight, scale, 19, 4)
    weights = SimpleNamespace(head=head, vocab_offset=0)
    adapter = ResidentHead(weights, SimpleNamespace(qmm=qmm))
    inputs = torch.randn(3, 4)
    actual = adapter.logits(inputs, 4, 11)
    expected = (inputs.bfloat16().float() @ weight[4:11].T).bfloat16().float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls[0]['SK'] == 1
    assert calls[0]['N'] == 7
    assert calls[0]['weight'].untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    assert calls[0]['scale'].untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
    assert adapter.owner is head


def test_identity_counts_unique_resident_storage_without_loading_weights():
    head_weight = torch.ones(3, 2)
    head = F8(head_weight, torch.ones(3, 1), 3, 2)
    weights = SimpleNamespace(world=2, rank=0, cfg=SimpleNamespace(streams=4),
                              layers=[head, head], head=head, norm=head_weight[0])
    engine = SimpleNamespace(w=weights)
    identity = resident_identity(engine)
    assert identity['engine_weights_object'] == id(weights)
    assert identity['storage_count'] == 2
    assert identity['resident_suffix_bytes'] == 36
    assert identity['checkpoint_readers_created'] == 0
    assert identity['decoded_matrix_cache_bytes'] == 0
    head_weight.requires_grad_(True)
    with pytest.raises(ValueError, match='gradients'):
        resident_identity(engine)


def test_resident_kda_replay_keeps_earlier_state_gradients_and_requires_fresh_history():
    torch.manual_seed(906)
    config = SimpleNamespace(conv=4, lower=-5., eps=1e-5)
    layer = SimpleNamespace(heads=1, conv=torch.randn(384, 4) * .1,
                            a_log=torch.zeros(1), dt_bias=torch.zeros(128),
                            norm=torch.ones(128), b_off=640)
    projections = (torch.randn(9, 641) * .1).requires_grad_()
    decay = (torch.randn(9, 128) * .1).requires_grad_()
    gate = (torch.randn(9, 128) * .1).requires_grad_()
    convolution = torch.zeros(3, 384)
    recurrent = torch.zeros(1, 128, 128)
    output = kda_reference(projections, decay, gate, convolution, recurrent, layer, config)
    gradients = torch.autograd.grad(output[-1].square().sum(), (projections, decay, gate))
    assert output.shape == (9, 128)
    assert torch.all(gradients[0][0, 128:384].reshape(2, 128).norm(dim=1) > 0)
    # Decay at row zero acts on the zero initial state. Later rows decay writes.
    assert torch.count_nonzero(gradients[1][0]) == 0
    assert gradients[1][1].norm() > 0
    assert gradients[2][-1].norm() > 0
    assert torch.count_nonzero(gradients[2][:-1]) == 0
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    with pytest.raises(ValueError, match='fresh'):
        kda_reference(projections, decay, gate, convolution, recurrent + 1, layer, config)
