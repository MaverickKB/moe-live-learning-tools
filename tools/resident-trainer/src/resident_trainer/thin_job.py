"""Small paired jobs scheduled inside the accepted serving weight owners."""

import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

from .answer import interleave_answers, load_rows
from .fit import AddedSharedNeurons
from .thin_budget import ThinPassBudget, available_bytes
from .thin_native import ResidentSuffix, native_operators
from .settings import TRAIN_LAYER, WATCH_LAYER, authority, checkpoints_dir, receipts_dir

KEY = 'resident_trainer.engine'


THIN_START = 29768        # 0x7448: trainer start op, carried in the sealed serving sequence
THIN_RESUME = 29769       # head finished a nested serving iteration; worker returns to the job
JOIN_TIMEOUT = 10.0       # seconds the head waits for the worker to join; then cancel the job only
STEP_TIMEOUT = 600.0      # seconds a trainer-channel exchange may wait inside a running job
CHANNEL_TAG = 'resident-trainer-v1'


def encode_id(identifier):
    """A 32-hex job identity as eight 16-bit int32-safe words."""
    return [int(identifier[i:i + 4], 16) for i in range(0, 32, 4)]


def decode_id(words):
    return ''.join('%04x' % int(word) for word in words)


def strip_start(message):
    """Split a message body into (ops without THIN_START, start payloads)."""
    body = list(message)
    kept, starts, index = [], [], 0
    while index < len(body):
        op, size = int(body[index]), int(body[index + 1])
        chunk = body[index:index + 2 + size]
        if len(chunk) != 2 + size:
            raise RuntimeError('Malformed serving message while scanning for trainer start')
        if op == THIN_START:
            starts.append(chunk[2:])
        elif op == THIN_RESUME:
            starts.append(None)
        else:
            kept += chunk
        index += 2 + size
    return kept, starts


class TrainerChannel:
    """Trainer-only exchange over the rendezvous store, never the chat all-gather.

    Every value is a tagged JSON record naming the job, the paired sequence,
    the exchange name and the sending rank; anything else is rejected.
    """

    def __init__(self, store, identifier, sequence, rank, world=2):
        self.store = store
        self.identifier = identifier
        self.sequence = int(sequence)
        self.rank = rank
        self.world = world

    def key(self, name, rank):
        return '/'.join(['resident-trainer', self.identifier, str(self.sequence), name, 'rank' + str(rank)])

    def put(self, name, value):
        record = {'tag': CHANNEL_TAG, 'job': self.identifier, 'seq': self.sequence,
                  'name': name, 'rank': self.rank, 'value': value}
        self.store.set(self.key(name, self.rank), json.dumps(record).encode())

    def take(self, name, rank, timeout):
        from datetime import timedelta
        key = self.key(name, rank)
        try:
            self.store.wait([key], timedelta(seconds=timeout))
        except Exception as error:
            raise TimeoutError('Trainer channel ' + name + ' from rank ' + str(rank)
                               + ' not received in ' + str(timeout) + 's') from error
        record = json.loads(self.store.get(key))
        expected = (CHANNEL_TAG, self.identifier, self.sequence, name, rank)
        actual = (record.get('tag'), record.get('job'), record.get('seq'), record.get('name'),
                  record.get('rank'))
        if actual != expected:
            raise RuntimeError('Trainer channel record mismatch: ' + str(actual))
        return record['value']

    def exchange(self, name, value, timeout=STEP_TIMEOUT):
        self.put(name, value)
        return [value if rank == self.rank else self.take(name, rank, timeout)
                for rank in range(self.world)]


def store_of(decoder):
    store = getattr(decoder.g.comm, 'store', None)
    if store is None:
        raise RuntimeError('Paired trainer requires the rendezvous store')
    return store


def attach(decoder, record):
    """Install the paired start protocol once on this rank (no received patching)."""
    if record['sid'] not in decoder.lanes:
        raise ValueError('Resident job admission lacks its waking request')
    # Other chats may be live: the job itself starts only once the pair is idle
    # (checked at every finish) and is announced in the sealed sequence.
    if getattr(decoder, '_thin_finish_installed', False):
        return
    decoder._thin_pending = {}
    decoder._thin_running = False
    decoder._thin_start = None
    if decoder.w.rank == 0:
        original_finish = decoder.finish

        def finish(done):
            original_finish(done)
            if (decoder._thin_pending and not decoder.lanes and decoder.broken is None
                    and not decoder._thin_running):
                sid = next(iter(decoder._thin_pending))
                lead(decoder, decoder._thin_pending.pop(sid))

        decoder.finish = finish
    else:
        original_apply = decoder.apply
        original_bell = decoder.g._await_bell

        def apply(message):
            kept, starts = strip_start(message)
            original_apply(kept)
            for payload in starts:
                if payload is None:
                    decoder._thin_resumed = True
                    continue
                # received still names this message; follow increments it after apply.
                decoder._thin_start = (int(payload[0]), decode_id(payload[1:]), decoder.received)

        def await_bell():
            start = decoder._thin_start
            if start is not None and not decoder._thin_running:
                decoder._thin_start = None
                join(decoder, *start)
                if not decoder.idle:
                    # The job's last THIN_RESUME left chats live: rank 0 is not
                    # idle and rings no doorbell; follow reads its next message.
                    return None
            return original_bell()

        decoder.apply = apply
        decoder.g._await_bell = await_bell
    decoder._thin_finish_installed = True


def pending_by_id(decoder, identifier):
    for sid, pending in list(decoder._thin_pending.items()):
        if pending['policy']['thin_job']['id'] == identifier:
            return decoder._thin_pending.pop(sid)
    return None


def write_receipt(rank, identifier, report):
    path = receipts_dir() / ('thin-job-' + identifier + '-rank' + str(rank) + '.json')
    path.write_text(json.dumps(report, indent=2) + '\n')


def cancel_receipt(decoder, identifier, reason, phases):
    phases.append(['cancelled', time.time()])
    report = {'id': identifier, 'rank': decoder.w.rank, 'status': 'cancelled', 'error': reason,
              'serving_restart': False, 'serving_affected': False, 'phases': phases}
    write_receipt(decoder.w.rank, identifier, report)
    return report


def native_constants():
    from tensorfold.families.glm5_next.cuda.multi import IDLE, SEAL_MOD
    return IDLE, SEAL_MOD


def lead(decoder, record, join_timeout=JOIN_TIMEOUT, runner=None):
    """Head: announce the job in the sealed sequence, then wait for the worker to join.

    Never starts alone: without the worker's matching confirmation only this
    trainer job is cancelled; serving continues.
    """
    runner = runner or run
    IDLE, SEAL_MOD = native_constants()
    identifier = record['policy']['thin_job']['id']
    sequence = getattr(decoder, '_thin_sequence', 0) + 1
    decoder._thin_sequence = sequence
    phases = [['announce', time.time()]]
    decoder._emit(THIN_START, [sequence, *encode_id(identifier)])
    decoder._emit(IDLE, [])
    decoder._flush()                      # rings the idle worker, seals as message `sent`
    decoder.idle = True
    message = (decoder.sent - 1) % SEAL_MOD
    channel = TrainerChannel(store_of(decoder), identifier, sequence, 0)
    ready = prepare(decoder, record)
    channel.put('join', {'message': message, 'ready': ready is None, 'error': ready})
    decoder._thin_running = True        # new chats wait in the scheduler queue meanwhile
    try:
        try:
            peer = channel.take('join', 1, join_timeout)
        except TimeoutError as error:
            channel.put('decision', 'cancel')
            return cancel_receipt(decoder, identifier, str(error), phases)
        phases.append(['worker_joined', time.time()])
        if peer['message'] != message or not peer['ready'] or ready is not None:
            channel.put('decision', 'cancel')
            return cancel_receipt(decoder, identifier, 'Paired admission failed: head message '
                                  + str(message) + ' ready ' + str(ready) + '; worker ' + str(peer),
                                  phases)
        channel.put('decision', 'go')
        phases.append(['both_in_job', time.time()])
        return runner(decoder, record, channel, phases)
    finally:
        decoder._thin_running = False


def join(decoder, sequence, identifier, message, decision_timeout=JOIN_TIMEOUT * 3, runner=None):
    """Worker: confirm the sealed message number, then follow the head's decision."""
    runner = runner or run
    phases = [['start_received', time.time()]]
    channel = TrainerChannel(store_of(decoder), identifier, sequence, 1)
    record = pending_by_id(decoder, identifier)
    ready = 'Worker lacks the admitted job record' if record is None else prepare(decoder, record)
    channel.put('join', {'message': message, 'ready': ready is None, 'error': ready})
    decoder._thin_running = True
    try:
        try:
            decision = channel.take('decision', 0, decision_timeout)
        except TimeoutError as error:
            return cancel_receipt(decoder, identifier, str(error), phases)
        if decision != 'go':
            return cancel_receipt(decoder, identifier, 'Head cancelled the job', phases)
        phases.append(['both_in_job', time.time()])
        return runner(decoder, record, channel, phases)
    finally:
        decoder._thin_running = False


def prepare(decoder, record):
    """Validate the sealed job locally before joining; returns an error or None."""
    try:
        load_job(decoder.w, record)
    except Exception as error:
        return str(error)
    return None


def load_job(weights, record):
    root = weights.meta[KEY]['root'] / 'resident-trainer'
    selected = record['policy']['thin_job']
    identifier = selected['id']
    if len(identifier) != 32 or any(letter not in '0123456789abcdef' for letter in identifier):
        raise ValueError('Unsafe thinner job identity')
    raw = (root / 'jobs' / (identifier + '.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected['sha256']:
        raise ValueError('Thinner job differs from its admitted seal')
    return root, identifier, json.loads(raw)


def serve_waiting(decoder, channel, counter):
    """Both ranks pause training at the same boundary for ordinary work.

    The decision travels on the trainer channel; the nested iteration then uses
    the ordinary sealed serving sequence exactly as native serving does.
    """
    weights = decoder.w
    if weights.rank == 0:
        scheduler = decoder.g.scheduler
        wanted = bool(decoder.live() or not scheduler.waiting.empty()
                      or scheduler.held is not None or scheduler.loading or scheduler.ready)
        channel.put('serve-' + str(counter), int(wanted))
    else:
        wanted = bool(channel.take('serve-' + str(counter), 0, STEP_TIMEOUT))
    if not wanted:
        return False
    torch.cuda.synchronize(weights.device)
    if weights.rank == 0:
        # One scheduler iteration may flush several sealed messages (rounds,
        # FINISH, IDLE). Close it with an explicit RESUME so the worker knows
        # exactly when to return to the job.
        decoder.g.scheduler._iteration()
        resume(decoder)
    else:
        decoder._thin_resumed = False
        while not decoder._thin_resumed:
            decoder.follow(once=True)
    return True


def resume(decoder):
    IDLE, _ = native_constants()
    decoder._emit(THIN_RESUME, [])
    idle = not decoder.lanes
    if idle:
        decoder._emit(IDLE, [])
    decoder._flush()
    if idle:
        decoder.idle = True


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


def frozen_sample(weights, limit=96, depth=3):
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

    for index in (TRAIN_LAYER, WATCH_LAYER):
        try:
            layer = weights.layers[index]
        except Exception:
            continue
        walk('layer' + str(index), layer, 0)
    return found


class Watch:
    """While a job runs: live phase file and periodic all-thread stack dumps for diagnosis."""

    def __init__(self, rank, identifier, phases, every=15.0):
        import faulthandler
        import threading
        self.faulthandler = faulthandler
        self.path = receipts_dir() / ('thin-live-' + identifier + '-rank' + str(rank) + '.log')
        self.stacks = receipts_dir() / ('thin-stacks-' + identifier + '-rank' + str(rank) + '.txt')
        self.phases = phases
        self.every = every
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, name='thin-watch', daemon=True)

    def note(self, text):
        with self.path.open('a') as stream:
            stream.write('%.3f %s\n' % (time.time(), text))

    def loop(self):
        while not self.stop.wait(self.every):
            try:
                self.note('heartbeat last_phase=' + str(self.phases[-1][0]))
                with self.stacks.open('a') as stream:
                    stream.write('==== %.3f last_phase=%s\n' % (time.time(), self.phases[-1][0]))
                    stream.flush()
                    self.faulthandler.dump_traceback(file=stream, all_threads=True)
            except Exception:
                pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        return False


def job_authorized(job, root):
    """A job runs only if it carries the configured operator authority and no STOP file exists.

    The authority is read on every guard check, so removing it (or touching STOP)
    aborts a running job at its next microbatch boundary.
    """
    expected = authority()
    allowed = bool(expected) and job.get('authority') == expected
    return allowed and not (Path(root) / 'STOP').exists()


def run(decoder, record, channel, phases):
    """Run one sealed microbatch group; failures return an owned job receipt."""
    from .thin import FRAME
    weights = decoder.w
    root, identifier, job = load_job(weights, record)
    report = {'id': identifier, 'rank': weights.rank, 'mode': job['mode'],
              'started_at': time.time(), 'private_ceiling_bytes': 2 * 1024**3,
              'free_memory_floor_bytes': 1_000_000_000, 'serving_restart': False,
              'sequence': channel.sequence, 'phases': phases, 'served_during_job': 0}

    watch = Watch(weights.rank, identifier, phases)

    def mark(name):
        torch.cuda.synchronize(weights.device)
        phases.append([name, time.time()])
        watch.note('phase ' + name + ' mem_available=' + str(available_bytes())
                   + ' cuda_allocated=' + str(torch.cuda.memory_allocated()))
        report['cuda_allocated_' + name] = torch.cuda.memory_allocated()
        report['mem_available_' + name] = available_bytes()
    budget = None
    scratch = None
    watch.__enter__()
    watch.note('job entered seq=' + str(channel.sequence) + ' mode=' + str(job['mode']))
    try:
        def authorized():
            return job_authorized(job, root)

        def cuda_bytes():
            return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

        with torch.inference_mode(False), torch.enable_grad():
            budget = ThinPassBudget(weights, authorized, job['deadline'], cuda_bytes,
                                    exchange=channel.exchange)
            frozen = frozen_sample(weights)
            report['frozen_tensors_hashed'] = [name for name, _ in frozen]
            report['frozen_bytes_hashed'] = sum(t.numel() * t.element_size() for _, t in frozen)
            report['frozen_sha256_before'] = fingerprint(frozen)
            FRAME.begin(channel.sequence)
            counter = [0]
            with budget:
                mark('budget_admitted')

                def guard():
                    budget.check()
                    decoder.iteration_since = time.monotonic()
                    counter[0] += 1
                    if serve_waiting(decoder, channel, counter[0]):
                        report['served_during_job'] += 1

                if job['mode'] == 'hotload':
                    from .hotload import hotload
                    report.update(hotload(__package__, Path(__file__).parent, job['modules'],
                                          channel.exchange))
                    if report['hotload'] != 'applied':
                        raise RuntimeError('Hot load refused: ' + json.dumps(report.get('verdicts')))
                    guard()
                    report['frozen_sha256_after'] = fingerprint(frozen)
                    report['frozen_unchanged'] = report['frozen_sha256_after'] == report['frozen_sha256_before']
                    mark('done')
                else:
                    if job.get('reset'):
                        weights.meta.pop('resident_trainer_model', None)
                    state = weights.meta.get('resident_trainer_model')
                    if state is None:
                        initial_name = str(job.get('initial', 'initial'))
                        if not initial_name.replace('-', '').isalnum():
                            raise ValueError('Unsafe initial checkpoint name')
                        path = root / (initial_name + '-rank' + str(weights.rank) + '.pt')
                        initial = torch.load(path, weights_only=True, map_location=weights.device)
                        hidden = weights.cfg.hidden
                        width = initial['model']['gate.weight'].shape[0]
                        model = AddedSharedNeurons(hidden, width, streams=4).to(weights.device)
                        model.load_state_dict(initial['model'])
                        optimizer = torch.optim.Adam([
                            {'params': [model.gate.weight, model.up.weight]},
                            {'params': [model.down.weight]}])
                        optimizer.load_state_dict(initial['optimizer'])
                        state = {'model': model, 'optimizer': optimizer,
                                 'completed': initial['completed'], 'cursor': initial['completed'] * 8}
                        state['orig_lrs'] = [g['lr'] for g in optimizer.param_groups]
                        state['reset_completed'] = initial['completed']
                        weights.meta['resident_trainer_model'] = state
                        del initial
                    report['added_sha256_before'] = fingerprint(sorted(state['model'].state_dict().items()))
                    before_params = [p.detach().clone() for p in state['model'].parameters()]
                    mark('state_loaded')
                    guard()
                    index_name = str(job.get('index', 'answer-index.json'))
                    if '/' in index_name or not index_name.endswith('.json'):
                        raise ValueError('Unsafe training index name')
                    index = json.loads((root / index_name).read_text())
                    report['index'] = index_name
                    fixed_ids = list(job.get('fixed_eval') or [])
                    fixed = {}
                    positive = []
                    negative = []
                    for example in index['examples']:
                        if example['id'] in fixed_ids:
                            fixed[example['id']] = example
                        elif example['positive']:
                            positive.append(example)
                        else:
                            negative.append(example)
                    positive = interleave_answers(positive)
                    if job['mode'] == 'probe':
                        positive.sort(key=lambda example: len(example['token_ids']))
                    batches = []
                    maximum = 0
                    for offset in range(job['microbatches']):
                        position = 0 if job['mode'] == 'probe' else state['cursor'] + offset
                        example = positive[position % len(positive)]
                        replay = negative[position % len(negative)]
                        maximum = max(maximum, len(example['token_ids']))
                        batches.append((example, replay))
                    if set(fixed) != set(fixed_ids):
                        raise ValueError('Fixed eval ids missing from index: ' + str(sorted(set(fixed_ids) - set(fixed))))
                    for example in fixed.values():
                        maximum = max(maximum, len(example['token_ids']))
                    from tensorfold.families.glm5_next.cuda.forward import Buffers
                    scratch = Buffers(weights, maximum, capacity=max(256, maximum), prefill=True)
                    suffix = ResidentSuffix(decoder.g, native_operators(), scratch, TRAIN_LAYER, guard)
                    report['resident_identity'] = suffix.identity
                    mark('batches_built')
                    if job['mode'] == 'probe':
                        example, replay = batches[0]
                        data = load_rows(root, example, TRAIN_LAYER, weights.device)
                        state['optimizer'].zero_grad(set_to_none=True)
                        answer, added = suffix.answer_loss(state['model'], data, example)
                        mark('forward_replay')
                        answer.backward()
                        mark('backward_replay')
                        norms = []
                        for parameter in state['model'].parameters():
                            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                                raise ValueError('Missing or nonfinite resident answer gradient')
                            norms.append(float(parameter.grad.norm()))
                        replay_gradients = []
                        for parameter in state['model'].parameters():
                            replay_gradients.append(parameter.grad.detach().clone())
                        replay_ce = float(answer.detach())
                        state['optimizer'].zero_grad(set_to_none=True)
                        del answer, added
                        eager, added = suffix.answer_loss(state['model'], data, example, use_replay=False)
                        eager.backward()
                        mark('eager_forward_backward')
                        errors = []
                        for reference, parameter in zip(replay_gradients, state['model'].parameters(), strict=True):
                            difference = (reference - parameter.grad).norm()
                            scale = reference.norm().clamp_min(1e-8)
                            errors.append(float(difference / scale))
                        if max(errors) > .001 or abs(float(eager.detach()) - replay_ce) > .0001:
                            raise ValueError('Resident eager/replay gradient qualification failed')
                        report['eager_replay_gradient_relative_errors'] = errors
                        report['eager_answer_ce'] = float(eager.detach())
                        report['gradient_qualified'] = True
                        report.update(answer_ce=replay_ce, gradient_norms=norms,
                                      example_id=example['id'], sequence_tokens=len(example['token_ids']),
                                      source_completed=state['completed'], updated=False)
                        state['optimizer'].zero_grad(set_to_none=True)
                        del eager, added, replay_gradients
                    elif job['mode'] == 'update':
                        basis_path = root / 'replay-basis.pt'
                        basis = torch.load(basis_path, weights_only=True, map_location=weights.device)
                        mark('basis_loaded')
                        if job.get('lr_scale') is not None:
                            scale = float(job['lr_scale'])
                            warm = int(job.get('warmup_steps') or 0)
                            done = state['completed'] - state['reset_completed']
                            factor = min(1.0, (done + 1) / warm) if warm else 1.0
                            for group, base in zip(state['optimizer'].param_groups, state['orig_lrs']):
                                group['lr'] = base * scale * factor
                        report['lrs'] = [g['lr'] for g in state['optimizer'].param_groups]
                        
                        def fixed_eval():
                            from .answer import load_rows as _load_rows
                            values = []
                            with torch.no_grad():
                                for name in fixed_ids:
                                    example = fixed[name]
                                    data = _load_rows(root, example, TRAIN_LAYER, weights.device)
                                    answer, _ = suffix.answer_loss(state['model'], data, example)
                                    values.append(float(answer))
                                    del data, answer
                                    guard()
                            return values

                        if fixed_ids:
                            before = fixed_eval()
                            report['fixed_eval_losses_before'] = before
                            report['fixed_eval_loss_before'] = sum(before) / len(before)
                        report.update(suffix.update(state['model'], state['optimizer'], root, batches, basis))
                        mark('optimizer_step')
                        if fixed_ids:
                            values = fixed_eval()
                            report['fixed_eval_ids'] = fixed_ids
                            report['fixed_eval_losses'] = values
                            report['fixed_eval_loss'] = sum(values) / len(values)
                            mark('fixed_eval')
                        state['cursor'] += len(batches)
                        state['completed'] += 1
                        report['updated'] = True
                        report['completed'] = state['completed']
                        path = checkpoints_dir() / ('thin-' + str(job.get('run', 'step')) + str(state['completed'])
                                                   + '-rank' + str(weights.rank) + '.pt')
                        saved = {'model': state['model'].state_dict(),
                                 'optimizer': state['optimizer'].state_dict(),
                                 'completed': state['completed'], 'cursor': state['cursor']}
                        torch.save(saved, path)
                        report['checkpoint_path'] = str(path)
                        mark('writeback')
                    else:
                        raise ValueError('Unknown resident job mode')
                    guard()
                    after = list(state['model'].parameters())
                    report['added_sha256_after'] = fingerprint(sorted(state['model'].state_dict().items()))
                    report['added_delta_norm'] = float(torch.sqrt(sum(((a.detach() - b) ** 2).sum()
                                                                      for a, b in zip(after, before_params))))
                    report['added_weights_changed'] = report['added_sha256_after'] != report['added_sha256_before']
                    report['frozen_sha256_after'] = fingerprint(frozen)
                    report['frozen_unchanged'] = report['frozen_sha256_after'] == report['frozen_sha256_before']
                    if not report['frozen_unchanged']:
                        raise RuntimeError('Frozen serving tensors changed during the trainer job')
                    mark('done')
                report['status'] = 'completed'
    except BaseException as error:
        report.update(status='aborted', error=str(error), traceback=traceback.format_exc())
        state = weights.meta.get('resident_trainer_model')
        if state is not None:
            state['optimizer'].zero_grad(set_to_none=True)
    finally:
        FRAME.end()
        watch.__exit__()
        watch.note('job exit status=' + str(report.get('status')) + ' error=' + str(report.get('error')))
        if budget is not None:
            report['memory'] = budget.receipt()
        report['seconds'] = time.time() - report['started_at']
        write_receipt(weights.rank, identifier, report)
    return report
