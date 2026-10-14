"""Bounded answer-loss and layer replay for a resident frozen backbone.

The owner supplies resident matrix views and coordinated TP collectives.
This module never opens a checkpoint or allocates a second backbone.
"""

import torch


class TensorHead:
    """CPU/reference head using a view of the owner's frozen matrix."""

    def __init__(self, weight, offset=0):
        if weight.ndim != 2 or weight.requires_grad:
            raise ValueError('Vocabulary matrix must be frozen [vocab, hidden]')
        self.weight = weight
        self.offset = offset
        self.rows, self.hidden = weight.shape

    def logits(self, inputs, start, stop):
        return inputs @ self.weight[start:stop].T

    def input_vjp(self, inputs, gradient, start, stop):
        return gradient @ self.weight[start:stop]


TRAINER_MAGIC = 0x7448494E  # 'THIN': marks a trainer collective, never a serving message


class TrainerFrame:
    """While active, every trainer collective is preceded by a tagged header.

    The header carries the magic, the paired job sequence, a per-job collective
    counter and the element count. A peer that is not inside the same job
    collective (for example a serving message) is detected and the trainer
    aborts instead of misreading the payload.
    """

    def __init__(self):
        self.job = None
        self.count = 0

    def begin(self, job):
        self.job = int(job)
        self.count = 0

    def end(self):
        self.job = None


FRAME = TrainerFrame()


class TrainerDesync(RuntimeError):
    """A trainer collective did not match its peer; only the trainer aborts."""


def check_header(rows):
    first = rows[0]
    for row in rows:
        if row[0] != TRAINER_MAGIC or list(row) != list(first):
            raise TrainerDesync('Trainer collective header mismatch: ' + str(rows))


def gather_values(value, weights):
    """Use only the existing pair communicator, in the owner's command order."""
    value = value.contiguous()
    if FRAME.job is not None:
        FRAME.count += 1
        header = torch.tensor([TRAINER_MAGIC, FRAME.job, FRAME.count, value.numel()],
                              device=value.device, dtype=torch.int64)
        rows = torch.empty((weights.world, 4), device=value.device, dtype=torch.int64)
        weights.comm.all_gather(header, rows.reshape(-1))
        check_header(rows.tolist())
    result = torch.empty((weights.world, *value.shape), device=value.device,
                         dtype=value.dtype)
    weights.comm.all_gather(value.reshape(-1), result.reshape(-1))
    return result


class ReplicatedInput(torch.autograd.Function):
    """A replicated input feeds distinct TP projections; sum their input VJPs."""

    @staticmethod
    def forward(context, inputs, weights):
        context.weights = weights
        return inputs.view_as(inputs)

    @staticmethod
    def backward(context, gradient):
        parts = gather_values(gradient, context.weights)
        return parts.sum(0), None


class GatherBranches(torch.autograd.Function):
    """Replicate rank branches forward; each rank owns its branch gradient."""

    @staticmethod
    def forward(context, partial, weights):
        context.rank = weights.rank
        return gather_values(partial, weights)

    @staticmethod
    def backward(context, gradient):
        return gradient[context.rank].contiguous(), None


def check_shape(inputs, targets, head, tile_rows):
    if inputs.ndim != 2 or inputs.shape[1] != head.hidden or not len(inputs):
        raise ValueError('Answer states must be nonempty [tokens, hidden]')
    if targets.shape != (len(inputs),) or targets.dtype != torch.long:
        raise ValueError('Every supervised state requires one integer target')
    if tile_rows <= 0 or head.rows <= 0 or head.offset < 0:
        raise ValueError('Vocabulary tiles and shard bounds must be positive')
    if not torch.isfinite(inputs).all():
        raise ValueError('Nonfinite supervised state')


class StreamedCrossEntropy(torch.autograd.Function):
    """Exact full-vocabulary CE in tiles, with a streamed frozen-head input VJP."""

    @staticmethod
    def forward(context, inputs, targets, head, tile_rows, weights, guard):
        check_shape(inputs, targets, head, tile_rows)
        maximum = torch.full((len(inputs),), -torch.inf, device=inputs.device,
                             dtype=inputs.dtype)
        exponential_sum = torch.zeros_like(maximum)
        target_logits = torch.zeros_like(maximum)
        target_count = torch.zeros_like(maximum)
        for start in range(0, head.rows, tile_rows):
            guard()
            stop = min(start + tile_rows, head.rows)
            logits = head.logits(inputs, start, stop).to(inputs.dtype)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite vocabulary logits')
            new_maximum = torch.maximum(maximum, logits.max(1).values)
            exponential_sum = exponential_sum * (maximum - new_maximum).exp()
            exponential_sum += (logits - new_maximum[:, None]).exp().sum(1)
            maximum = new_maximum
            local_targets = targets - head.offset - start
            selected = (local_targets >= 0) & (local_targets < stop - start)
            positions = torch.where(selected)[0]
            target_logits[positions] = logits[positions, local_targets[positions]]
            target_count += selected.to(inputs.dtype)
            del logits
            guard()
        if weights is not None:
            parts = gather_values(torch.stack((maximum, exponential_sum,
                                               target_logits, target_count)), weights)
            global_maximum = parts[:, 0].max(0).values
            exponential_sum = (parts[:, 1] * (parts[:, 0] - global_maximum).exp()).sum(0)
            maximum = global_maximum
            target_logits = parts[:, 2].sum(0)
            target_count = parts[:, 3].sum(0)
        if not torch.equal(target_count, torch.ones_like(target_count)):
            raise ValueError('Targets must occur exactly once across vocabulary shards')
        log_normalizer = maximum + exponential_sum.log()
        context.save_for_backward(inputs, targets, log_normalizer)
        context.head = head
        context.tile_rows = tile_rows
        context.weights = weights
        context.guard = guard
        return (log_normalizer - target_logits).mean()

    @staticmethod
    def backward(context, gradient):
        inputs, targets, log_normalizer = context.saved_tensors
        head = context.head
        input_gradient = torch.zeros_like(inputs)
        for start in range(0, head.rows, context.tile_rows):
            context.guard()
            stop = min(start + context.tile_rows, head.rows)
            logits = head.logits(inputs, start, stop).to(inputs.dtype)
            probabilities = (logits - log_normalizer[:, None]).exp()
            local_targets = targets - head.offset - start
            selected = (local_targets >= 0) & (local_targets < stop - start)
            positions = torch.where(selected)[0]
            probabilities[positions, local_targets[positions]] -= 1
            probabilities /= len(inputs)
            input_gradient += head.input_vjp(inputs, probabilities, start, stop)
            del logits, probabilities
            context.guard()
        if context.weights is not None:
            input_gradient = gather_values(input_gradient, context.weights).sum(0)
        return input_gradient * gradient, None, None, None, None, None


def streamed_answer_loss(inputs, targets, head, tile_rows=4096, weights=None,
                         guard=lambda: None):
    """Include every vocabulary row while projecting supervised positions only."""
    return StreamedCrossEntropy.apply(inputs, targets, head, tile_rows, weights, guard)


class LayerReplay(torch.autograd.Function):
    """Retain four-stream boundaries and build one frozen layer's graph at a time."""

    @staticmethod
    def forward(context, inputs, layers, guard):
        if not layers:
            raise ValueError('Middle training requires a frozen downstream layer span')
        boundaries = []
        current = inputs
        for layer in layers:
            guard()
            boundaries.append(current)
            current = layer(current)
            guard()
        context.save_for_backward(*boundaries)
        context.layers = layers
        context.guard = guard
        return current

    @staticmethod
    def backward(context, gradient):
        boundaries = context.saved_tensors
        for index in range(len(context.layers) - 1, -1, -1):
            context.guard()
            with torch.enable_grad():
                inputs = boundaries[index].detach().requires_grad_(True)
                output = context.layers[index](inputs)
                gradient = torch.autograd.grad(output, inputs, gradient)[0]
            del inputs, output
            context.guard()
        return gradient, None, None


def replay_suffix(inputs, layers, guard=lambda: None):
    return LayerReplay.apply(inputs, tuple(layers), guard)


def accumulated_update(model, optimizer, examples, loss_for, guard=lambda: None,
                       project=None, clip=5, weights=None, project_weights=None):
    """B1 accumulation preserves the existing mean-per-example B8 objective."""
    if not examples:
        raise ValueError('An update requires admitted training examples')
    optimizer.zero_grad(set_to_none=True)
    losses = []
    try:
        for example in examples:
            guard()
            loss = loss_for(example)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('Answer loss must be a finite scalar')
            losses.append(float(loss.detach()))
            (loss / len(examples)).backward()
            del loss
            guard()
        for parameter in model.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError('Missing or nonfinite added-neuron gradient')
        if project is not None:
            project(model)
        if weights is None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            squared = torch.zeros((), device=weights.device)
            for parameter in model.parameters():
                squared += parameter.grad.float().square().sum()
            norm = gather_values(squared, weights).sum().sqrt()
            multiplier = (clip / (norm + 1e-6)).clamp(max=1)
            for parameter in model.parameters():
                parameter.grad.mul_(multiplier)
        guard()
        optimizer.step()
        if project_weights is not None:
            with torch.no_grad():
                project_weights(model)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {'microbatch': 1, 'examples': len(examples),
            'loss': sum(losses) / len(losses), 'losses': losses, 'gradient_norm': float(norm)}


def shard_added_checkpoint(state, rank):
    """Partition only the saved added neurons and their Adam moments."""
    if rank not in (0, 1):
        raise ValueError('Added checkpoint requires one of the two serving ranks')
    names = ('gate.weight', 'up.weight', 'down.weight')
    if set(state['model']) != set(names):
        raise ValueError('Checkpoint must contain only the added shared neurons')
    width, hidden = state['model']['gate.weight'].shape
    if width % 2 or state['model']['up.weight'].shape != (width, hidden):
        raise ValueError('Added width must split equally across the pair')
    if state['model']['down.weight'].shape != (hidden * 4, width):
        raise ValueError('Saved addition must have four-stream down weights')
    parameter_ids = []
    groups = []
    for source_group in state['optimizer']['param_groups']:
        group = dict(source_group)
        group['params'] = list(source_group['params'])
        parameter_ids.extend(group['params'])
        groups.append(group)
    if len(parameter_ids) != 3 or len(set(parameter_ids)) != 3:
        raise ValueError('Saved optimizer must own exactly the three added parameters')
    first = rank * (width // 2)
    model = {}
    moments = {}
    for name, parameter_id in zip(names, parameter_ids, strict=True):
        axis = 1 if name == 'down.weight' else 0
        value = state['model'][name]
        model[name] = value.narrow(axis, first, width // 2).clone()
        local = {}
        for field, tensor in state['optimizer']['state'][parameter_id].items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim:
                if tensor.shape != value.shape:
                    raise ValueError('Saved Adam moment shape differs from its added parameter')
                local[field] = tensor.narrow(axis, first, width // 2).clone()
            elif isinstance(tensor, torch.Tensor):
                local[field] = tensor.clone()
            else:
                local[field] = tensor
        moments[parameter_id] = local
    return {'model': model, 'optimizer': {'state': moments, 'param_groups': groups},
            'completed': state['completed'], 'rank': rank, 'world': 2}
