# SPDX-License-Identifier: Apache-2.0
"""Resident native forwards and frozen input VJPs, with straight-through casts.

Adapted from the origin project's native operators. The owning trainer supplies
private scratch. Native parity is a required measurement before fitting.
"""

import torch
from tensorfold.cuda.kernels import qmm as packed_qmm
from tensorfold.families.glm5_next.cuda import glue, kda, kv8, latent, qmm
from torch.nn import functional as F

from .resident_math import (
    constant,
    hc_post,
    hc_reference,
    kda_reference,
    native_value,
    rms_reference,
    rotate,
    rounded,
)


def decode_dense(q, start=0, stop=None):
    stop = q.n if stop is None else stop
    if isinstance(q, qmm.Q4):
        words, scales, biases = packed_qmm.unpack(q)
        shifts = torch.arange(8, device=words.device) * 4
        digits = (words[start:stop, :, None].long() >> shifts & 15).float()
        digits = digits.reshape(stop - start, q.k // q.gs, q.gs)
        result = digits * scales[start:stop, :, None].float()
        result = result + biases[start:stop, :, None].float()
        return result.reshape(stop - start, q.k).to(torch.bfloat16).float()
    if isinstance(q, qmm.F8):
        result = q.weight[start:stop].float().reshape(stop - start, q.k // 128, 128)
        return (result * q.scale[start:stop, :, None]).reshape(stop - start, q.k)
    return q.weight[start:stop].float().clone()


class Dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, prompt, f32):
        ctx.q = q
        xb = x.to(torch.bfloat16).contiguous()
        if prompt and isinstance(q, qmm.Q4):
            result = packed_qmm.prefill_matmul(xb, q, f32=f32)
        else:
            result = qmm.matmul(xb, q, f32=f32)
        return result.float().clone()

    @staticmethod
    def backward(ctx, grad):
        q = ctx.q
        dx = torch.zeros((grad.shape[0], q.k), device=grad.device)
        for start in range(0, q.n, 2048):
            stop = min(start + 2048, q.n)
            weight = decode_dense(q, start, stop)
            dx.add_(grad[:, start:stop].float() @ weight)
            del weight
        return (dx, None, None, None)


def linear(x, q, prompt=True, f32=False):
    return Dense.apply(x, q, prompt, f32)


class RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.x = x.detach()
        ctx.weight = weight
        ctx.eps = eps
        xb = x.to(torch.bfloat16).contiguous()
        output = torch.empty_like(xb)
        glue.rmsnorm(xb, weight, eps, output)
        return output.float().clone()

    @staticmethod
    def backward(ctx, grad):
        with torch.enable_grad():
            x = ctx.x.detach().requires_grad_()
            output = rms_reference(x, ctx.weight, ctx.eps)
            dx = torch.autograd.grad(output, x, grad)[0]
        return (dx, None, None)


class HCPre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, norm, c):
        ctx.x, ctx.h, ctx.norm, ctx.c = (x.detach(), h, norm, c)
        count = x.shape[0]
        xb = x.to(torch.bfloat16).reshape(count, -1).contiguous()
        normed = torch.empty((count, c.hidden), device=x.device, dtype=torch.bfloat16)
        sums = torch.empty((count, c.hidden // 64), device=x.device)
        post = torch.empty((count, 4), device=x.device)
        comb = torch.empty((count, 16), device=x.device)
        partial = torch.empty((count, glue.HC_BLOCKS, 32), device=x.device)
        glue.hc_pre(
            xb,
            h.fn,
            h.base,
            h.scale,
            norm,
            normed,
            sums,
            post,
            comb,
            partial,
            c.eps,
            c.hc_eps,
            c.hc_iters,
            prompt=True,
        )
        return (normed.float().clone(), post.clone(), comb.reshape(count, 4, 4).clone())

    @staticmethod
    def backward(ctx, gn, gp, gc):
        with torch.enable_grad():
            x = ctx.x.detach().requires_grad_()
            outputs = hc_reference(x, ctx.h, ctx.norm, ctx.c)
            dx = torch.autograd.grad(outputs, x, (gn, gp, gc))[0]
        return (dx, None, None, None)


def trellis_matrix(ex, expert, kind):
    import os

    os.environ.setdefault("MAX_JOBS", "2")
    from tensorfold.cuda.exl3 import experts as x3

    trellis = getattr(ex, kind + "t")[expert].contiguous().view(torch.int16)
    return x3.dequant(trellis, "mcg")


class TrellisLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ex, expert, kind):
        ctx.ex, ctx.expert, ctx.kind = (ex, expert, kind)
        suh = constant(getattr(ex, "suh_" + kind)[expert])
        svh = constant(getattr(ex, "svh_" + kind)[expert])
        matrix = trellis_matrix(ex, expert, kind)
        rotated = rounded(rotate(x * suh), torch.float16)
        output = rotated @ matrix.float()
        return rounded(rotate(output) * svh, torch.float16)

    @staticmethod
    def backward(ctx, grad):
        ex, expert, kind = (ctx.ex, ctx.expert, ctx.kind)
        suh = constant(getattr(ex, "suh_" + kind)[expert])
        svh = constant(getattr(ex, "svh_" + kind)[expert])
        matrix = trellis_matrix(ex, expert, kind)
        dx = rotate(grad * svh) @ matrix.float().T
        return (rotate(dx) * suh, None, None, None)


class Experts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, picks, ex, b, limit):
        from tensorfold.cuda import experts as grouped
        from tensorfold.families.glm5_next.cuda import exl3_mm

        count = x.shape[0]
        ctx.x, ctx.picks, ctx.ex, ctx.limit = (x.detach(), picks.clone(), ex, limit)
        grouped.route(picks, b.plan, b.plan.tile)
        out = torch.empty((count, b.top_k + 1, ex.dims), device=x.device)
        exl3_mm.routed(
            x.to(torch.bfloat16).contiguous(),
            picks,
            b.plan,
            ex,
            b.exl3,
            out.view(-1, ex.dims),
            count,
            limit,
        )
        return out[:, : b.top_k].clone()

    @staticmethod
    def backward(ctx, grad):
        dx = torch.zeros_like(ctx.x)
        picks = ctx.picks[:, : grad.shape[1]]
        for expert in torch.unique(picks).tolist():
            rows, slots = torch.where(picks == expert)
            with torch.enable_grad():
                x = ctx.x.index_select(0, rows).detach().requires_grad_()
                gate = TrellisLinear.apply(x, ctx.ex, expert, "g").clamp(max=ctx.limit)
                up = TrellisLinear.apply(x, ctx.ex, expert, "u").clamp(
                    -ctx.limit, ctx.limit
                )
                act = rounded(rounded(F.silu(gate), torch.float16) * up, torch.float16)
                output = TrellisLinear.apply(act, ctx.ex, expert, "d")
                contribution = torch.autograd.grad(output, x, grad[rows, slots])[0]
            dx.index_add_(0, rows, contribution)
        return (dx, None, None, None, None, None)


def moe(x, layer, w, b, delta=None):
    c = w.cfg
    m = layer.moe
    logits = x @ constant(m.router).T
    native_logits = torch.empty_like(logits)
    glue.router(x.detach().to(torch.bfloat16).contiguous(), m.router, native_logits)
    logits = native_value(logits, native_logits)
    scores = torch.sigmoid(logits)
    with torch.no_grad():
        all_picks = torch.empty(
            (x.shape[0], c.top_k + 1), device=x.device, dtype=torch.int32
        )
        native_weights = torch.empty_like(all_picks, dtype=torch.float32)
        glue.select(
            native_logits,
            m.bias,
            all_picks,
            native_weights,
            c.top_k,
            c.experts,
            c.routed_scale,
            c.norm_topk,
        )
        picks = all_picks[:, : c.top_k].long()
    weights = scores.gather(1, picks)
    if c.norm_topk:
        weights = weights / (weights.sum(1, keepdim=True) + 1e-20)
    weights = weights * c.routed_scale
    weights = native_value(weights, native_weights[:, : c.top_k])
    expert_outputs = Experts.apply(x, all_picks, m.experts, b, c.limit)
    gu = linear(x, m.shared.gu)
    act = swiglu(gu, c.limit)
    branch = linear(act, m.shared.down, f32=True)
    if delta is not None:
        branch = branch + delta
    for slot in range(c.top_k):
        if slot == 0:
            routed = expert_outputs[:, slot] * weights[:, slot, None]
        else:
            routed = routed + expert_outputs[:, slot] * weights[:, slot, None]
    result = routed + branch
    native_rows = torch.empty((x.shape[0], c.top_k + 1, c.hidden), device=x.device)
    native_rows[:, : c.top_k].copy_(expert_outputs.detach())
    native_rows[:, c.top_k].zero_()
    native_result = torch.empty_like(result)
    glue.combine_shared(
        native_rows, branch.detach().contiguous(), native_weights, native_result
    )
    return native_value(result, native_result)


class KDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, a, g, conv, rec, k, c, position):
        ctx.p, ctx.a, ctx.g, ctx.conv, ctx.rec = (
            p.detach(),
            a.detach(),
            g.detach(),
            conv.detach(),
            rec.detach(),
        )
        ctx.k, ctx.c = (k, c)
        count = p.shape[0]
        scratch = kda.KDAScratch(count, k.heads, p.device)
        state_out = torch.empty_like(rec)
        out = kda.chain(
            p.to(torch.bfloat16).contiguous(),
            k.b_off,
            a.to(torch.bfloat16).contiguous(),
            g.to(torch.bfloat16).contiguous(),
            conv.to(torch.bfloat16).contiguous(),
            k.conv,
            rec,
            k.a_log,
            k.dt_bias,
            k.norm,
            c.eps,
            c.lower,
            count,
            scratch,
            state_out,
            wide=False,
            pos=position,
        )
        return out.float().clone()

    @staticmethod
    def backward(ctx, grad):
        with torch.enable_grad():
            p = ctx.p.detach().requires_grad_()
            a = ctx.a.detach().requires_grad_()
            g = ctx.g.detach().requires_grad_()
            out = kda_reference(p, a, g, ctx.conv, ctx.rec, ctx.k, ctx.c)
            dp, da, dg = torch.autograd.grad(out, (p, a, g), grad)
        return (dp, da, dg, None, None, None, None, None)


def absorb_weights(a, kind):
    if isinstance(a, latent.AbsorbQ4):
        words = getattr(a, "w" + kind + "w")
        scales = getattr(a, "w" + kind + "s")
        biases = getattr(a, "w" + kind + "b")
        shifts = torch.arange(8, device=words.device) * 4
        decoded = (words[:, :, :, None].long() >> shifts & 15).float()
        decoded = decoded.reshape(words.shape[0], words.shape[1], -1, 64)
        return (decoded * scales[:, :, :, None] + biases[:, :, :, None]).reshape(
            words.shape[0], words.shape[1], -1
        )
    weight = getattr(a, "w" + kind).float()
    scale = getattr(a, "s" + kind, None)
    if scale is not None:
        shape = weight.shape
        weight = (
            weight.reshape(shape[0], shape[1], -1, 128) * scale[:, :, :, None]
        ).reshape(shape)
    return weight.clone()


def dsa(x, layer, w, b, prefix, nch=None, trace=None):
    a, c = (layer.dsa, w.cfg)
    projections = linear(x, a.proj)
    if trace is not None:
        trace["projection"] = projections
    qr = RMS.apply(projections[:, : c.q_lora], a.q_norm, c.eps)
    history = RMS.apply(projections[:, c.q_lora :], a.kv_norm, c.eps)
    if trace is not None:
        trace["qr"] = qr
        trace["latent"] = history
    quantized = kv8.dequantize(
        kv8.quantize_rows(history.detach().to(torch.bfloat16))
    ).float()
    history = native_value(history, quantized)
    query = linear(qr, a.q_b).reshape(-1, a.heads, c.qk_dim)
    if trace is not None:
        trace["query"] = query
    key = absorb_weights(a.absorb, "k")
    qa = torch.einsum("rhd,hdk->rhk", query, key)
    query_out = torch.empty(qa.shape, device=x.device, dtype=b.lat_s.qa.dtype)
    native = latent.absorb_q(
        query.detach().to(torch.bfloat16).contiguous(),
        a.absorb,
        query_out,
        prompt=latent.PROMPT_MMA,
    ).float()
    qa = native_value(qa, native)
    if trace is not None:
        trace["qa"] = qa
    cache = torch.cat((prefix, history), 0)
    score = torch.einsum("rhd,nd->rhn", qa, cache) * c.qk_dim ** (-0.5)
    start = prefix.shape[0]
    mask = (
        torch.arange(cache.shape[0], device=x.device)[None, :]
        <= start + torch.arange(x.shape[0], device=x.device)[:, None]
    )
    score = score.masked_fill(~mask[:, None, :], -torch.inf)
    readout = torch.einsum("rhn,nd->rhd", torch.softmax(score, -1), cache)
    from tensorfold.families.glm5_next.cuda.forward import dense_attention

    packed = kv8.quantize_rows(cache.detach().to(torch.bfloat16))
    position = torch.tensor([start], device=x.device, dtype=torch.int32)
    native_out = torch.empty(readout.shape, device=x.device, dtype=b.lat_s.ol.dtype)
    dense_attention(
        qa.detach().to(b.lat_s.qa.dtype).contiguous(),
        packed,
        position,
        b.lat_s,
        scale=c.qk_dim ** (-0.5),
        nch=nch or b.lat_s.nch,
        out=native_out,
        hb=latent.head_block(x.shape[0]),
        starts=b.att_starts,
    )
    readout = native_value(readout, native_out)
    if trace is not None:
        trace["readout"] = readout
        trace["packed"] = packed
    value = absorb_weights(a.absorb, "v")
    expanded = torch.einsum("rhk,hdk->rhd", readout, value)
    value_out = torch.empty(expanded.shape, device=x.device, dtype=torch.bfloat16)
    native = latent.expand_v(
        readout.detach().to(b.lat_s.ol.dtype).contiguous(),
        a.absorb,
        value_out,
        prompt=latent.PROMPT_MMA,
    ).float()
    expanded = native_value(expanded, native)
    return linear(expanded.reshape(x.shape[0], -1), a.o, f32=True)


class HCPost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branches, post, comb):
        ctx.x, ctx.branches, ctx.post, ctx.comb = (
            x.detach(),
            branches.detach(),
            post.detach(),
            comb.detach(),
        )
        count = x.shape[0]
        xb = x.to(torch.bfloat16).reshape(count, -1).contiguous()
        output = torch.empty_like(xb)
        glue.hc_post(
            xb,
            output,
            branches.contiguous(),
            post.contiguous(),
            comb.reshape(count, 16).contiguous(),
        )
        return output.reshape(count, 4, -1).float().clone()

    @staticmethod
    def backward(ctx, grad):
        with torch.enable_grad():
            x = ctx.x.detach().requires_grad_()
            branches = ctx.branches.detach().requires_grad_()
            post = ctx.post.detach().requires_grad_()
            comb = ctx.comb.detach().requires_grad_()
            out = hc_post(x, branches, post, comb)
            values = torch.autograd.grad(out, (x, branches, post, comb), grad)
        return values


def native_hc_post(x, branches, post, comb):
    return HCPost.apply(x, branches, post, comb)


def swiglu(gu, limit):
    half = gu.shape[1] // 2
    gate = gu[:, :half].clamp(max=limit)
    up = gu[:, half:].clamp(-limit, limit)
    reference = rounded(rounded(F.silu(gate)) * up)
    actual = torch.empty((gu.shape[0], half), device=gu.device, dtype=torch.bfloat16)
    sums = torch.empty((gu.shape[0], half // 64), device=gu.device)
    glue.swiglu(gu.detach().to(torch.bfloat16).contiguous(), actual, sums, limit)
    return native_value(reference, actual)

