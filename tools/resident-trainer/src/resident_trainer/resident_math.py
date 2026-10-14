# SPDX-License-Identifier: Apache-2.0
"""CPU-testable four-stream and causal adjoints for resident native forwards."""

import math

import torch
from torch.nn import functional as F


class NativeValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, actual):
        return actual.detach().float().clone()

    @staticmethod
    def backward(ctx, grad):
        return (grad, None)


def native_value(reference, actual):
    return NativeValue.apply(reference, actual)


def rounded(x, dtype=torch.bfloat16):
    return native_value(x, x.to(dtype).float())


def constant(x):
    return x.detach().float().clone()


def rms_reference(x, weight, eps):
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return rounded(rounded(x * inv) * constant(weight))


def hc_reference(x, h, norm, c):
    flat = x.reshape(x.shape[0], -1)
    inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + c.eps)
    mixing = flat @ constant(h.fn).T * inv
    base = constant(h.base)
    scale = constant(h.scale)
    pre = torch.sigmoid(mixing[:, :4] * scale[0] + base[:4]) + c.hc_eps
    post = 2 * torch.sigmoid(mixing[:, 4:8] * scale[1] + base[4:8])
    logits = (mixing[:, 8:24] * scale[2] + base[8:24]).reshape(-1, 4, 4)
    comb = torch.softmax(logits, dim=-1) + c.hc_eps
    comb = comb / (comb.sum(1, keepdim=True) + c.hc_eps)
    for _ in range(c.hc_iters - 1):
        comb = comb / (comb.sum(2, keepdim=True) + c.hc_eps)
        comb = comb / (comb.sum(1, keepdim=True) + c.hc_eps)
    collapsed = pre[:, 0, None] * x[:, 0] + pre[:, 1, None] * x[:, 1]
    collapsed = collapsed + pre[:, 2, None] * x[:, 2]
    collapsed = collapsed + pre[:, 3, None] * x[:, 3]
    normed = rms_reference(rounded(collapsed), norm, c.eps)
    return (normed, post, comb)


def hc_post(x, branches, post, comb):
    branch = branches[0]
    for rank in range(1, branches.shape[0]):
        branch = branch + branches[rank]
    branch = rounded(branch)
    streams = []
    for s in range(4):
        mixed = comb[:, 0, s, None] * x[:, 0] + comb[:, 1, s, None] * x[:, 1]
        mixed = mixed + comb[:, 2, s, None] * x[:, 2]
        mixed = mixed + comb[:, 3, s, None] * x[:, 3]
        streams.append(rounded(mixed + post[:, s, None] * branch))
    return torch.stack(streams, 1)


def rotate(x):
    shape = x.shape
    block = x.reshape(-1, 128)
    width = 1
    while width < 128:
        pairs = block.reshape(-1, 128 // (2 * width), 2, width)
        left = pairs[:, :, 0]
        right = pairs[:, :, 1]
        block = torch.stack((left + right, left - right), 2).reshape(-1, 128)
        width *= 2
    return (block / math.sqrt(128)).reshape(shape)


def kda_reference(p, a, g, conv, rec, k, c):
    """Rematerialize causal WY chunks instead of retaining one state per token."""
    from .reference import kda_chunks

    heads = k.heads
    channels = heads * 128
    count = len(p)
    history = torch.cat((conv, p[:, : 3 * channels]), 0)
    weights = constant(k.conv)
    convolved = history[:count] * weights[:, 0]
    for tap in range(1, c.conv):
        convolved = convolved + history[tap : tap + count] * weights[:, tap]
    values = rounded(F.silu(convolved)).reshape(count, 3, heads, 128)
    query = values[:, 0]
    key = values[:, 1]
    value = values[:, 2]
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-06)
    query = query / math.sqrt(128)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-06)
    decay_input = a.reshape(count, heads, 128) + constant(k.dt_bias).reshape(heads, 128)
    log_decay = c.lower * torch.sigmoid(
        torch.exp(constant(k.a_log))[:, None] * decay_input
    )
    beta = rounded(torch.sigmoid(p[:, k.b_off : k.b_off + heads]))
    if torch.count_nonzero(rec):
        raise ValueError("Thinner complete-sequence KDA requires a fresh causal state")
    output, _ = kda_chunks(query, key, value, log_decay, beta)
    output = rounded(output)
    output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + c.eps)
    output = output * constant(k.norm) * torch.sigmoid(g.reshape(count, heads, 128))
    return rounded(output).reshape(count, channels)
