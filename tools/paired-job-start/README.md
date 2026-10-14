# paired-job-start

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Running a side job (training, capture, reload) inside a live two-rank tensor-parallel server without desynchronising the serving collectives or blocking chats.

## Core idea

- The head appends a `THIN_START` op to the sealed serving message stream, so the worker sees it at the same message number.
- Both ranks confirm the message number and readiness on a trainer-only key-value channel before any work; otherwise only the job is cancelled.
- Between job steps both ranks agree whether to serve waiting chats; the head ends each nested serving iteration with an explicit `THIN_RESUME` op.
- After the job, the worker skips its idle doorbell wait if chats are still live.

## Invariants

- Never split the ranks: both run the job or neither does.
- Both ranks enter at the same serving message.
- Job traffic never uses the serving collectives.
- A failed join cancels the job, never serving.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- The decoder attributes (`_emit`, `_flush`, `apply`, `finish`, `g._await_bell`, `sent`, `received`, `lanes`, `idle`) mirror the origin engine.
- The op codes and the store come from that engine (PyTorch rendezvous store).

## Adapting it

1. Bind the module hooks: `native_constants`, `prepare`, `run`, `synchronize`, `write_receipt`.
2. Map the decoder surface to your engine; `tests/test_protocol.py` has a complete fake you can follow.
3. Choose op codes that cannot collide with your engine's serving ops.
4. Pick timeouts: `JOIN_TIMEOUT` should be short (cancel fast); `STEP_TIMEOUT` long enough for a microbatch.

## Untested or limited

- World sizes other than two.
- Engines whose serving loop does not seal messages in order.
- Pipeline parallelism.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```

See `LESSONS.md` for the failures behind this tool.
