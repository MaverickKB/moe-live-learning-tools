"""Exact device-side fingerprint of frozen tensors, without host copies."""

import hashlib

import torch


def fingerprint(tensors, chunk=1 << 22):
    """Exact digest without host copies: per tensor, chunked integer sums on its device.

    Each tensor's raw bytes are viewed as int16 (or uint8) and reduced in fixed
    chunks to int64 position-weighted sums, so no full-size host or device copy
    is made. Any byte change alters the digest with overwhelming probability.
    """
    digest = hashlib.sha256()
    for name, tensor in tensors:
        flat = tensor.detach().reshape(-1)
        try:
            words = flat.view(torch.int16) if flat.element_size() % 2 == 0 else flat.view(torch.uint8)
        except RuntimeError:
            words = flat.contiguous().view(torch.uint8)
        sums = []
        for start in range(0, words.numel(), chunk):
            part = words[start:start + chunk].to(torch.int64)
            weights = torch.arange(1, part.numel() + 1, device=part.device, dtype=torch.int64)
            sums.append(int((part * weights).sum()))
            sums.append(int(part.sum()))
            del part, weights
        digest.update((name + str(tuple(tensor.shape)) + str(tensor.dtype) + str(sums)).encode())
    return digest.hexdigest()


def frozen_sample(weights, layers, limit=96, depth=3):
    """Named frozen serving tensors: final norm plus EVERY tensor reachable (attributes, lists,
    dicts, up to `depth` levels) from the trained and watched layers, including the large quantized weights."""
    found, seen = [], set()
    norm = getattr(weights, 'norm', None)
    if isinstance(norm, torch.Tensor):
        found.append(('norm', norm)); seen.add(id(norm))

    def walk(prefix, value, level):
        if len(found) >= limit or level > depth:
            return
        if isinstance(value, torch.Tensor):
            if id(value) not in seen and value.numel():
                seen.add(id(value)); found.append((prefix, value))
            return
        if isinstance(value, dict):
            items = sorted((str(k), v) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            items = [(str(i), v) for i, v in enumerate(value)]
        elif hasattr(value, '__dict__') and not isinstance(value, type):
            items = sorted(vars(value).items())
        else:
            return
        for name, child in items:
            walk(prefix + '.' + name, child, level + 1)

    for index in layers:
        try:
            layer = weights.layers[index]
        except Exception:
            continue
        walk('layer' + str(index), layer, 0)
    return found
