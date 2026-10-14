"""Resident TensorFold adapter, invoked by the serving pair's scheduler.

The owning scheduler supplies the already-loaded engine, native input-VJP
operators, private short-sequence scratch and a paired memory/deadline guard.
Native kernel and gradient parity require a separate qualification receipt.
"""

import os
from dataclasses import replace

import torch

from .thin import (
    GatherBranches,
    ReplicatedInput,
    accumulated_update,
    replay_suffix,
    streamed_answer_loss,
)


def native_operators():
    """Import the source-local backend only inside an authorized native lease."""
    from .engine import resident_ops

    return resident_ops


class ResidentHead:
    """Vocabulary tiles view the rank's existing FP8/BF16 classifier storage."""

    def __init__(self, weights, operators):
        self.owner = weights.head
        self.weights = weights
        self.operators = operators
        self.rows = self.owner.n
        self.hidden = self.owner.k
        self.offset = weights.vocab_offset
        # Packed Q4 rows have a tiled layout. The observed classifier is F8.
        if type(self.owner).__name__ not in ('F8', 'B16'):
            raise ValueError('Resident classifier must have direct FP8/BF16 row views')
        if self.owner.weight.requires_grad:
            raise ValueError('Resident classifier must remain frozen')

    def logits(self, inputs, start, stop):
        fields = {'weight': self.owner.weight[start:stop], 'n': stop - start}
        if hasattr(self.owner, 'scale'):
            fields['scale'] = self.owner.scale[start:stop]
        tile = replace(self.owner, **fields)
        values = inputs.to(torch.bfloat16).contiguous()
        return resident_head_tile(values, self.owner, tile, self.operators.qmm)

    def input_vjp(self, inputs, gradient, start, stop):
        weight = self.operators.decode_dense(self.owner, start, stop)
        result = gradient.float() @ weight
        del weight
        return result.to(inputs.dtype)


def resident_head_tile(inputs, owner, tile, qmm):
    """Keep the original classifier's K-reduction order when its rows are tiled."""
    rows = len(inputs)
    split_k = qmm.b16_split_k(owner.n, owner.k)
    block_m = qmm.bucket(min(rows, 128))
    fp8 = type(owner).__name__ == 'F8'
    config = qmm.F8_CONFIG if fp8 else qmm.B16_CONFIG
    warps, stages = config[block_m]
    block_n = qmm.F8_BN_DECODE if fp8 and block_m == 16 else qmm.B16_BN
    sequential = qmm._seq(split_k, rows, owner.n, owner.k)
    prompt_config = qmm._prompt_cfg(split_k, rows, owner.n, owner.k)
    if prompt_config is not None:
        block_m, block_n, warps, stages, sequential = prompt_config
    output = torch.empty((rows, tile.n), device=inputs.device, dtype=torch.bfloat16)
    partial = output
    if split_k > 1 and not sequential:
        partial = torch.empty((split_k * rows * tile.n,), device=inputs.device)
    grid_m = (rows + block_m - 1) // block_m
    grid_n = (tile.n + block_n - 1) // block_n
    grid = (grid_m, grid_n, 1 if sequential else split_k)
    arguments = [inputs, tile.weight]
    if fp8:
        arguments.append(tile.scale)
    arguments.extend([output, partial, rows, inputs.stride(0)])
    kernel = qmm._fmm if fp8 else qmm._bmm
    kernel[grid](*arguments, N=tile.n, K=tile.k, SK=split_k, BM=block_m,
                 BLOCK_N=block_n, BK=qmm.B16_BK, F32=False, SEQ=sequential,
                 num_warps=warps, num_stages=stages)
    if split_k > 1 and not sequential:
        count = rows * tile.n
        qmm._reduce[((count + 1023) // 1024,)](partial, output, count, SK=split_k,
                                             BLOCK=1024, F32=False, num_warps=4)
    return output.float()


def resident_identity(engine):
    """Same-process object and storage proof, without reading traffic or tensors."""
    weights = engine.w
    if weights.world != 2 or weights.rank not in (0, 1) or weights.cfg.streams != 4:
        raise ValueError('Resident training requires the intact four-stream TP2 pair')
    storages = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise ValueError('Resident base tensor has gradients enabled')
            storage = value.untyped_storage()
            pointer = storage.data_ptr()
            storages[pointer] = storage.nbytes()
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, '__dataclass_fields__'):
            for name in value.__dataclass_fields__:
                visit(getattr(value, name))

    visit(weights.layers)
    visit(weights.head)
    visit(weights.norm)
    return {'pid': os.getpid(), 'rank': weights.rank, 'world': weights.world,
            'engine_weights_object': id(weights), 'storage_count': len(storages),
            'resident_suffix_bytes': sum(storages.values()),
            'checkpoint_readers_created': 0, 'decoded_matrix_cache_bytes': 0}


class ResidentSuffix:
    """Full causal suffix over private states and borrowed packed rank weights."""

    def __init__(self, engine, operators, scratch, middle_layer, guard):
        self.identity = resident_identity(engine)
        self.engine = engine
        self.weights = engine.w
        self.operators = operators
        self.scratch = scratch
        self.middle_layer = middle_layer
        self.guard = guard
        if middle_layer not in (11, 23, 35):
            raise ValueError('Shared knowledge must use the approved DSA placement')
        if scratch is engine.e.pbuf or scratch is engine.e.buf:
            raise ValueError('Training requires private scratch beside serving')
        if not scratch.prefill or scratch.split is not None:
            raise ValueError('Resident trainer requires private unsplit prompt scratch')
        self.head = ResidentHead(self.weights, operators)

    def layer(self, streams, index):
        w = self.weights
        c = w.cfg
        layer = w.layers[index]
        ops = self.operators
        b = self.scratch
        if len(streams) > b.rows or len(streams) > c.dense_limit:
            raise ValueError('Complete example exceeds private dense-attention scratch')
        normed, post, combine = ops.HCPre.apply(streams, layer.attn_hc, layer.in_norm, c)
        local_inputs = ReplicatedInput.apply(normed, w)
        if layer.kind == 'dsa':
            prefix = torch.empty((0, c.kv_lora), device=streams.device, dtype=torch.float32)
            partial = ops.dsa(local_inputs, layer, w, b, prefix)
        else:
            k = layer.kda
            projections = ops.linear(local_inputs, k.proj)
            a = ops.linear(projections[:, k.fa_off:k.fa_off + 128], k.fb)
            g = ops.linear(projections[:, k.ga_off:k.ga_off + 128], k.gb)
            convolution = torch.zeros((c.conv - 1, 3 * k.heads * 128), device=streams.device)
            recurrent = torch.zeros((k.heads, 128, 128), device=streams.device)
            values = ops.KDA.apply(projections, a, g, convolution, recurrent, k, c, 0)
            partial = ops.linear(values, k.o, f32=True)
        branches = GatherBranches.apply(partial, w)
        streams = ops.native_hc_post(streams, branches, post, combine)
        normed, post, combine = ops.HCPre.apply(streams, layer.ffn_hc, layer.post_norm, c)
        local_inputs = ReplicatedInput.apply(normed, w)
        if layer.moe is None:
            raise ValueError('Approved middle suffix must retain native shared/routed MoE')
        partial = ops.moe(local_inputs, layer, w, b)
        branches = GatherBranches.apply(partial, w)
        return ops.native_hc_post(streams, branches, post, combine)

    def answer_loss(self, model, data, example, use_replay=True):
        """Train local added neurons through all later frozen layers and the head."""
        self.guard()
        if self.engine.w is not self.weights:
            raise ValueError('Resident weight owner changed during the training lease')
        w = self.weights
        ops = self.operators
        added = model(data['input'].float())
        if added.shape != data['before'].shape:
            raise ValueError('Added shared neurons must write across all four streams')
        branches = GatherBranches.apply(added, w).sum(0)
        before = data['before'].float()
        outputs = []
        for stream in range(4):
            branch = data['base_output'].float() + branches[:, stream]
            mixed = ops.native_hc_post(before, branch.unsqueeze(0), data['post'], data['combine'])
            outputs.append(mixed[:, stream])
        streams = torch.stack(outputs, dim=1)
        layers = []
        for index in range(self.middle_layer + 1, w.cfg.layers):
            def run_layer(values, index=index):
                return self.layer(values, index)
            layers.append(run_layer)
        if use_replay:
            streams = replay_suffix(streams, layers, self.guard)
        else:
            for layer in layers:
                self.guard()
                streams = layer(streams)
        targets = torch.tensor(example['target_ids'], device=streams.device, dtype=torch.long)
        start = example['answer_start']
        if start < 0 or start + len(targets) > len(streams):
            raise ValueError('Supervised answer positions escaped the complete sequence')
        streams = streams[start:start + len(targets)]
        hidden = streams[:, 0] + streams[:, 1]
        hidden = hidden + streams[:, 2]
        hidden = ops.rounded((hidden + streams[:, 3]) / 4)
        normed = ops.RMS.apply(hidden, w.norm, w.cfg.eps)
        answer = streamed_answer_loss(normed, targets, self.head, weights=w, guard=self.guard)
        return answer, added

    def update(self, model, optimizer, root, batches, basis=None):
        """Preserve Study09's answer, prefix and replay objective under B1 TP."""
        from .answer import load_rows, remove_replay_component

        def loss_for(batch):
            example, replay_example = batch
            data = load_rows(root, example, self.middle_layer, self.weights.device)
            replay = load_rows(root, replay_example, self.middle_layer, self.weights.device)
            answer, added = self.answer_loss(model, data, example)
            scale = data['base_output'].float().square().mean().clamp_min(1e-4)
            preservation = model(replay['input'].float()).square().mean() / scale
            prefix = added[:example['answer_start']].square().mean() / scale
            return answer + 10 * preservation + 5 * prefix

        def project_gradients(added):
            if basis is not None:
                with torch.no_grad():
                    remove_replay_component(added.gate.weight.grad, basis)
                    remove_replay_component(added.up.weight.grad, basis)

        def project_weights(added):
            if basis is not None:
                remove_replay_component(added.gate.weight, basis)
                remove_replay_component(added.up.weight, basis)

        with torch.inference_mode(False), torch.enable_grad():
            return accumulated_update(model, optimizer, batches, loss_for, self.guard,
                                      project_gradients, weights=self.weights,
                                      project_weights=project_weights)
