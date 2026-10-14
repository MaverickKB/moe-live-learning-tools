# AGENTS.md: paired-job-start

Use this when you need two tensor-parallel ranks to start, pause and finish a side job in lockstep inside live serving.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/paired_jobs/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.

## Must hold

- Never split the ranks: both run the job or neither does.
- Both ranks enter at the same serving message.
- Job traffic never uses the serving collectives.
- A failed join cancels the job, never serving.

## To adapt

1. Bind the module hooks: `native_constants`, `prepare`, `run`, `synchronize`, `write_receipt`.
2. Map the decoder surface to your engine; `tests/test_protocol.py` has a complete fake you can follow.
3. Choose op codes that cannot collide with your engine's serving ops.
4. Pick timeouts: `JOIN_TIMEOUT` should be short (cancel fast); `STEP_TIMEOUT` long enough for a microbatch.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
