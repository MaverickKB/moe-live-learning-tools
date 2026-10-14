"""Request-owned resident trainer extension for the accepted native hook ABI.

Registration occurs through the existing lazy native-gradient module. Frozen
weights remain in their serving process. Ordinary requests use the original
callbacks; private jobs execute after their waking request releases its slot.
"""

import hashlib
import json
from pathlib import Path

from ..settings import receipts_dir

from tensorfold_host import native  # host-engine module, not shipped here (see engine/README.md)

KEY = 'resident_trainer.engine'
REGISTERED = False
ORIGINALS = {}


def own_record(record):
    if record is None:
        return False
    policy = record['policy']
    return bool(policy.get('neurons') or policy.get('neuron_capture') or policy.get('thin_job'))


def row_count(buffers):
    count = 0
    for state, offset, size in getattr(buffers, '_eval_pieces', ()):
        count = max(count, offset + size)
    return count


def complete_ffn(weights, buffers):
    pending = getattr(buffers, '_thin_ffn_pending', None)
    if pending is None:
        return False
    index, count = pending
    buffers._thin_ffn_pending = None
    gathered = buffers.gath[:weights.world * count * weights.cfg.hidden]
    gathered = gathered.reshape(weights.world, count, weights.cfg.hidden)
    native.after_ffn(weights.layers[index], weights, buffers, count, gathered)
    return index == 44


def register():
    """Compose explicit hooks once; no tensor or serving lifecycle changes."""
    global REGISTERED
    if REGISTERED:
        return
    from tensorfold_host import attention_native, eval_native, round2_native

    originals = {'install': attention_native.install, 'shared': eval_native.shared,
                 'begin': round2_native.begin_layer, 'loss': eval_native.loss,
                 'finished': round2_native.finished}

    def install(decoder, record):
        originals['install'](decoder, record)
        if not own_record(record):
            return
        selected = record['policy'].get('neurons')
        if selected is not None:
            native.artifact(decoder.w, selected)
        record['thin_decoder'] = decoder
        if record['policy'].get('thin_job'):
            from ..thin_job import attach
            attach(decoder, record)

    def shared(layer, weights, buffers, first, stop):
        result = originals['shared'](layer, weights, buffers, first, stop)
        controlled = False
        for record, state, offset, size in native.records(weights, buffers):
            if own_record(record):
                controlled = True
                break
        if not controlled:
            return result
        count = row_count(buffers)
        if first == 0:
            native.before_ffn(layer, weights, buffers, count)
            buffers._thin_ffn_pending = (layer.index, count)
        native.shared(layer, weights, buffers, first, stop)
        return result

    def begin(layer, weights, buffers, count):
        complete_ffn(weights, buffers)
        return originals['begin'](layer, weights, buffers, count)

    def loss(weights, buffers, count):
        if complete_ffn(weights, buffers):
            from tensorfold.families.glm5_next.cuda import glue
            glue.stream_mean(buffers.x[:count], buffers.hidden[:count])
            glue.rmsnorm(buffers.hidden[:count], weights.norm, weights.cfg.eps,
                         buffers.fnormed[:count], buffers.fxs[:count])
        return originals['loss'](weights, buffers, count)

    def finished(weights, record):
        result = originals['finished'](weights, record)
        if not own_record(record):
            return result
        result.update(native.finished(weights, record))
        if record['policy'].get('thin_job'):
            decoder = record['thin_decoder']
            decoder._thin_pending[record['sid']] = record
        return result

    # All callbacks are constructed before the five assignments.
    attention_native.install = install
    eval_native.shared = shared
    round2_native.begin_layer = begin
    eval_native.loss = loss
    round2_native.finished = finished
    ORIGINALS.update(originals)
    REGISTERED = True
    rank = eval_native.STATE.rank
    receipt = {'rank': rank, 'extension': 'resident_thinner_v1', 'registered': True,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'serving_restart': False, 'second_weight_reader': False}
    path = receipts_dir() / ('thin-runtime-registered-rank' + str(rank) + '.json')
    path.write_text(json.dumps(receipt, indent=2) + '\n')
