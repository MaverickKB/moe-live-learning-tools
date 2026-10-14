"""CPU toy: train added shared-expert neurons at a middle layer of a frozen model.

This mirrors the shape of the resident trainer without any engine, GPU or
second rank: a frozen stack of layers, a small set of added neurons inserted
at one middle layer, the full-vocabulary streamed cross-entropy, the per-layer
replay of the frozen suffix, and the frozen-weight fingerprint taken before
and after training.

    python examples/toy_train.py
"""

import torch
from torch import nn

from resident_trainer.fit import AddedSharedNeurons
from resident_trainer.thin import TensorHead, accumulated_update, replay_suffix, streamed_answer_loss
from resident_trainer.thin_job import fingerprint

HIDDEN, VOCAB, WIDTH, LAYERS, MIDDLE = 32, 101, 8, 6, 3


class FrozenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return x + torch.tanh(self.proj(x))


def main(steps=60, seed=0):
    torch.manual_seed(seed)
    blocks = nn.ModuleList(FrozenBlock() for _ in range(LAYERS))
    embed = torch.randn(VOCAB, HIDDEN)
    unembed = torch.randn(VOCAB, HIDDEN) / HIDDEN ** 0.5
    for parameter in blocks.parameters():
        parameter.requires_grad_(False)
    frozen = [(name, p) for name, p in blocks.named_parameters()] + [('embed', embed), ('unembed', unembed)]
    before = fingerprint(frozen)

    head = TensorHead(unembed)
    added = AddedSharedNeurons(HIDDEN, WIDTH)
    optimizer = torch.optim.Adam(added.parameters(), lr=3e-2)

    # Four "facts": a prompt token whose answer token the frozen model does not produce.
    prompts = torch.tensor([3, 17, 42, 77])
    answers = torch.tensor([9, 55, 12, 90])

    def prefix(tokens):
        x = embed[tokens]
        with torch.no_grad():
            for block in blocks[:MIDDLE]:
                x = block(x)
        return x

    def loss_for(index):
        x = prefix(prompts[index:index + 1])
        x = x + added(x)
        x = replay_suffix(x, blocks[MIDDLE:])
        return streamed_answer_loss(x, answers[index:index + 1], head, tile_rows=16)

    def accuracy():
        with torch.no_grad():
            x = prefix(prompts)
            x = x + added(x)
            for block in blocks[MIDDLE:]:
                x = block(x)
            return int(((x @ unembed.T).argmax(1) == answers).sum())

    start = accuracy()
    for _ in range(steps):
        report = accumulated_update(added, optimizer, list(range(len(prompts))), loss_for)
    after = fingerprint(frozen)
    result = {'correct_before': start, 'correct_after': accuracy(), 'total': len(prompts),
              'final_loss': round(report['loss'], 4), 'frozen_unchanged': before == after}
    print(result)
    return result


if __name__ == '__main__':
    main()
