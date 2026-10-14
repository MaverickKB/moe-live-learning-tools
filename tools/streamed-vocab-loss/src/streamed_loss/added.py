"""The only trainable parameters: an added gate/up/down block on a frozen base."""

from torch import nn
from torch.nn import functional as F


class AddedSharedNeurons(nn.Module):
    """Added gate/up/down neurons in one frozen-base shared-expert pocket."""

    def __init__(self, hidden, width, streams=1):
        super().__init__()
        if streams not in (1, 4):
            raise ValueError('Shared output requires one or four residual streams')
        self.streams = streams
        self.gate = nn.Linear(hidden, width, bias=False)
        self.up = nn.Linear(hidden, width, bias=False)
        self.down = nn.Linear(width, hidden * streams, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, inputs):
        gate = self.gate(inputs).clamp(max=10)
        up = self.up(inputs).clamp(-10, 10)
        activations = F.silu(gate) * up
        output = self.down(activations)
        if self.streams == 4:
            output = output.reshape(*inputs.shape[:-1], 4, inputs.shape[-1])
        return output
