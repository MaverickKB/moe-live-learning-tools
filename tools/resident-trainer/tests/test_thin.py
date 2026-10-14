"""Prove bounded loss, full suffix gradients and replicated TP adjoints on CPU."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from resident_trainer.fit import AddedSharedNeurons
from resident_trainer.thin import (
    GatherBranches,
    ReplicatedInput,
    TensorHead,
    accumulated_update,
    replay_suffix,
    shard_added_checkpoint,
    streamed_answer_loss,
)


class CpuCollective:
    """Two CPU threads model the existing ranks' ordered all-gather calls."""

    def __init__(self):
        self.condition = threading.Condition()
        self.calls = {}

    def rank(self, rank):
        owner = self

        class Rank:
            index = 0

            def all_gather(self, inputs, output):
                with owner.condition:
                    values = owner.calls.setdefault(self.index, {})
                    values[rank] = inputs.clone()
                    owner.condition.notify_all()
                    ready = owner.condition.wait_for(lambda: len(values) == 2, timeout=10)
                    if not ready:
                        raise RuntimeError('CPU pair collective diverged')
                    output.copy_(torch.cat((values[0], values[1])))
                    self.index += 1

        return SimpleNamespace(rank=rank, world=2, device='cpu', comm=Rank())


def run_pair(operation):
    collective = CpuCollective()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = []
        for rank in range(2):
            results.append(pool.submit(operation, collective.rank(rank)))
        return [result.result(timeout=30) for result in results]


@pytest.mark.parametrize('tile_rows', [1, 7, 64])
def test_streamed_full_vocabulary_loss_and_vjp_match_dense(tile_rows):
    torch.manual_seed(901)
    inputs = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(19, 4, dtype=torch.float64) * 30
    targets = torch.tensor([0, 18, 2, 7, 13])
    expected = F.cross_entropy(inputs @ weight.T, targets)
    expected_gradient = torch.autograd.grad(expected, inputs)[0]
    actual = streamed_answer_loss(inputs, targets, TensorHead(weight), tile_rows)
    actual_gradient = torch.autograd.grad(actual, inputs)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert weight.grad is None


def test_streamed_tp_head_combines_both_denominators_and_gradients_once():
    torch.manual_seed(902)
    states = torch.randn(5, 4, dtype=torch.float64)
    weight = torch.randn(19, 4, dtype=torch.float64)
    targets = torch.tensor([0, 18, 2, 7, 13])
    expected_states = states.clone().requires_grad_()
    expected = F.cross_entropy(expected_states @ weight.T, targets)
    expected_gradient = torch.autograd.grad(expected, expected_states)[0]

    def operation(weights):
        start, stop = (0, 9) if weights.rank == 0 else (9, 19)
        inputs = states.clone().requires_grad_()
        head = TensorHead(weight[start:stop], offset=start)
        actual = streamed_answer_loss(inputs, targets, head, 4, weights)
        actual_gradient = torch.autograd.grad(actual, inputs)[0]
        return actual, actual_gradient

    for actual, actual_gradient in run_pair(operation):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert weight.grad is None


def test_tp_gradients_match_unsharded_network_with_residual_and_two_layers():
    torch.manual_seed(903)
    inputs = torch.randn(3, 4, dtype=torch.float64)
    matrices = torch.randn(2, 2, 4, 4, dtype=torch.float64) * .1
    expected_inputs = inputs.clone().requires_grad_()
    current = expected_inputs
    for index in range(2):
        total = matrices[index, 0] + matrices[index, 1]
        current = current + torch.sin(current @ total.T)
    expected_gradient = torch.autograd.grad(current.square().sum(), expected_inputs)[0]

    def operation(weights):
        local_inputs = inputs.clone().requires_grad_()
        current = local_inputs
        for index in range(2):
            branch_inputs = ReplicatedInput.apply(current, weights)
            partial = branch_inputs @ matrices[index, weights.rank].T
            branches = GatherBranches.apply(partial, weights)
            current = current + torch.sin(branches.sum(0))
        actual_gradient = torch.autograd.grad(current.square().sum(), local_inputs)[0]
        return current, actual_gradient

    for actual, actual_gradient in run_pair(operation):
        torch.testing.assert_close(actual, current, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_twenty_layer_causal_replay_preserves_all_four_stream_gradients():
    torch.manual_seed(904)
    inputs = torch.randn(6, 4, 3, dtype=torch.float64, requires_grad=True)
    matrix = torch.randn(4, 4, dtype=torch.float64) * .02

    def layer(streams):
        history = streams.cumsum(0) / torch.arange(1, 7)[:, None, None]
        mixed = torch.einsum('rsh,st->rth', history, matrix)
        return streams + mixed.tanh()

    current = inputs
    layers = [layer] * 20
    for operation in layers:
        current = operation(current)
    objective = current[-1].square().sum()
    expected_gradient = torch.autograd.grad(objective, inputs)[0]
    actual = replay_suffix(inputs, layers)
    actual_gradient = torch.autograd.grad(actual[-1].square().sum(), inputs)[0]
    torch.testing.assert_close(actual, current, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert torch.all(actual_gradient[0].norm(dim=1) > 0)


def test_microbatch_one_matches_mean_per_example_adam_update():
    torch.manual_seed(905)
    model = torch.nn.Linear(4, 3).double()
    expected_model = copy.deepcopy(model)
    examples = [torch.randn(count, 4, dtype=torch.float64) for count in [2, 5, 3, 4]]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    expected_optimizer = torch.optim.Adam(expected_model.parameters(), lr=.001)
    expected_losses = []
    for inputs in examples:
        expected_losses.append(expected_model(inputs).square().mean())
    torch.stack(expected_losses).mean().backward()
    torch.nn.utils.clip_grad_norm_(expected_model.parameters(), 5)
    expected_optimizer.step()
    report = accumulated_update(model, optimizer, examples,
                                lambda inputs: model(inputs).square().mean())
    assert report['microbatch'] == 1
    for actual, expected in zip(model.parameters(), expected_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_guard_aborts_pending_update_without_changing_neurons():
    model = torch.nn.Linear(2, 2)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters())
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('Pair headroom breached')

    with pytest.raises(RuntimeError, match='headroom'):
        accumulated_update(model, optimizer, [torch.ones(1, 2)] * 2,
                           lambda inputs: model(inputs).square().mean(), guard)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, initial[name])
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())


def test_streamed_loss_rejects_missing_target_and_trainable_base():
    with pytest.raises(ValueError, match='frozen'):
        TensorHead(torch.ones(3, 2, requires_grad=True))
    with pytest.raises(ValueError, match='exactly once'):
        streamed_answer_loss(torch.ones(1, 2), torch.tensor([3]), TensorHead(torch.ones(3, 2)))


def test_sharded_adam_continuation_and_global_clip_match_the_saved_full_pocket():
    torch.manual_seed(907)
    expected = AddedSharedNeurons(4, 6, streams=4)
    optimizer = torch.optim.Adam([
        {'params': [expected.gate.weight, expected.up.weight], 'lr': .02},
        {'params': [expected.down.weight], 'lr': .01}])
    for parameter in expected.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    source = {'model': copy.deepcopy(expected.state_dict()),
              'optimizer': copy.deepcopy(optimizer.state_dict()), 'completed': 4}
    gradients = {}
    for name, parameter in expected.named_parameters():
        gradients[name] = torch.randn_like(parameter)
        parameter.grad = gradients[name].clone()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), .5)
    optimizer.step()

    def operation(weights):
        shard = shard_added_checkpoint(source, weights.rank)
        model = AddedSharedNeurons(4, 3, streams=4)
        model.load_state_dict(shard['model'])
        local_optimizer = torch.optim.Adam([
            {'params': [model.gate.weight, model.up.weight], 'lr': .02},
            {'params': [model.down.weight], 'lr': .01}])
        local_optimizer.load_state_dict(shard['optimizer'])
        local_gradients = {}
        for name, gradient in gradients.items():
            axis = 1 if name == 'down.weight' else 0
            local_gradients[name] = gradient.narrow(axis, weights.rank * 3, 3)

        def loss_for(example):
            loss = torch.zeros(())
            for name, parameter in model.named_parameters():
                loss = loss + (parameter * local_gradients[name]).sum()
            return loss

        accumulated_update(model, local_optimizer, [None, None], loss_for,
                           clip=.5, weights=weights)
        return model.state_dict()

    states = run_pair(operation)
    for name, parameter in expected.named_parameters():
        axis = 1 if name == 'down.weight' else 0
        actual = torch.cat((states[0][name], states[1][name]), dim=axis)
        torch.testing.assert_close(actual, parameter, rtol=1e-6, atol=1e-8)
