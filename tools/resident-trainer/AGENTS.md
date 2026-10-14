# AGENTS.md: resident-trainer

Guide for coding agents working in this tool. Read `README.md` for status and results and
`LESSONS.md` before changing any coordination code.

## Map

```
src/resident_trainer/
  thin.py            Engine-free autograd: streamed full-vocab CE, layer replay, gradient
                     accumulation, rank gather helpers, checkpoint sharding. Start here.
  fit.py             AddedSharedNeurons (the only trainable parameters) and offline fitting.
  thin_budget.py     ThinPassBudget: free-memory floor, workspace ceiling, deadline, authority;
                     paired abort decisions. Raises PassAborted (trainer-only).
  thin_job.py        Live protocol: THIN_START / THIN_RESUME ops, TrainerChannel, lead/join,
                     attach (installs the start hooks), serve_waiting, fingerprint, run.
  thin_native.py     Adapter from the engine's resident ops to the autograd in thin.py.
  answer.py          Loading per-rank answer rows and training examples.
  reference.py       Differentiable reference suffix used for parity checks.
  resident_math.py   Pure-torch reference math used by the native ops and tests.
  scope.py, files.py Input scope checks and hashing helpers.
  hotload.py         Paired no-restart code loading (hash, agree, exec exact bytes).
  pair_recovery.py   Single-owner locked recovery.
  recall_scoring.py  Offline strict scoring and sibling-value leak detection.
  settings.py        Environment settings with generic defaults.
  engine/            Engine-specific reference code (not importable here, not tested).
tests/               CPU tests; fake decoders and stores stand in for the server.
examples/toy_train.py  Minimal CPU end-to-end of the training math.
```

## Invariants (do not break these)

1. **Never split the ranks.** Both ranks run the same job, the same microbatches and the
   same boundaries, or neither does. A one-rank job deadlocks the serving collectives.
2. **Both ranks enter a job at the same serving message.** The start is carried inside the
   sealed serving sequence and confirmed by message number before any work. Never start a
   job from a side channel alone.
3. **Trainer traffic stays off the serving collectives.** Use `TrainerChannel` (or a
   trainer-tagged collective framed by `TrainerFrame`), never the chat all-gather.
4. **Base weights stay frozen.** Only `AddedSharedNeurons` parameters receive gradients.
   The fingerprint must match before and after every job; a mismatch is a hard failure.
5. **The memory floor aborts only the trainer.** `PassAborted` discards the pending update.
   It must never stop, restart or reconfigure serving.
6. **Serving keeps priority.** Waiting chats are served at every guard boundary.
7. **One recovery owner.** Every restart path takes the `pair_recovery` lock.
8. **Hot-load only what is safe.** Files in `hotload.RESTART_ONLY` need a planned restart.

## Adapting to another model or engine

1. Get `pytest` green, then run `examples/toy_train.py` to see the training math.
2. Implement the engine side, using `engine/` as a reference only:
   - a head object with `logits(inputs, start, stop)` and `input_vjp(...)` over your frozen
     classifier shard (see `thin.TensorHead`);
   - frozen layer callables for the suffix after your chosen layer (for `replay_suffix`);
   - capture of the residual input at the chosen layer for the training prompts;
   - a hook point after a request finishes (head) and before the worker blocks on its
     next message (worker), which is where `thin_job.attach` patches in.
3. Map the serving protocol: you need a way to append an op to the sealed message stream
   (`THIN_START`, `THIN_RESUME`) and a key-value store both ranks can reach.
4. Change the hard-coded shape assumptions: four residual streams, world size two, layer
   numbers (`settings.py`), allowed middle layers in `thin_native` and `answer`.
5. Pick the frozen tensors to fingerprint for your model (`frozen_sample`).
6. Keep the memory floor; tune `FLOOR_BYTES` and `WORKSPACE_BYTES` for your hosts.
7. Verify on your hardware: eager versus replay gradient parity (`probe` mode does this),
   then one update job, then controls.

## Pitfalls

- A worker that reads only one of several messages flushed by one scheduler iteration
  will be out of step. Close nested serving with an explicit marker (see LESSONS 2).
- Do not create optimizer state from a probe path; it changes the learning-rate base.
- Register every hook before the first job and check registration in the receipt.
- `importlib.reload` can execute stale bytecode. Hash and exec the exact bytes.
- Capturing many layers at once can push the host over its memory or swap limits.
- `frozen_sample` caps the tensor count; make sure the large quantised weights are
  inside the cap for your model.
