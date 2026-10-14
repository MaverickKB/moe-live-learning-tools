# resident-trainer

Train a small set of **added shared-expert neurons** at one middle layer of a large
mixture-of-experts model **inside the live serving process**, reusing the weights the
server already has loaded, with no model downtime.

> **Teaching example, not a product.** Adapt the ideas; do not copy it 1:1.
>
> **Status: reference extract, not a maintained product.** This code was pulled out of a
> research project so that people (and their coding agents) can read it, learn from it and
> adapt it. It ran on one specific setup. It has not been run anywhere else. Expect to
> rewrite the engine integration for your own stack.

## What it does

- The base model stays frozen. Training touches only an added gate/up/down block
  (`fit.AddedSharedNeurons`) whose output is added to the residual stream at one layer.
- The loss is exact full-vocabulary cross-entropy, computed in vocabulary tiles over the
  server's existing classifier storage (`thin.StreamedCrossEntropy`), and the frozen
  suffix of layers is replayed one layer at a time in the backward pass
  (`thin.LayerReplay`) so no full activation graph is kept.
- Jobs are small (a few microbatches) and are scheduled between chat requests. Between
  microbatches, both ranks pause together and serve any waiting chats.

## How it fits into a live tensor-parallel server

| Concern | Mechanism | Where |
|---|---|---|
| Both ranks start the same job at the same serving message | Paired sealed start: the head writes a `THIN_START` op into the sealed serving sequence; the worker confirms the message number on a trainer-only channel before anything runs. If the worker does not join in time, only the job is cancelled. | `thin_job.lead`, `thin_job.join`, `thin_job.attach` |
| Trainer traffic never mixes with serving collectives | Trainer-only channel over the rendezvous key-value store, every record tagged with job, sequence, name and rank | `thin_job.TrainerChannel` |
| Serving chats during a job | Both ranks agree at each boundary whether to serve; the head closes the nested iteration with an explicit `THIN_RESUME` marker | `thin_job.serve_waiting`, `thin_job.resume` |
| No stall after a job | Worker skips its doorbell wait if the job's last resume left chats live | `attach.await_bell` |
| Memory safety | Free-memory floor (default 1 GB) and private workspace ceiling sampled continuously; abort decisions are exchanged so both ranks abort the **trainer only** at the same boundary | `thin_budget.ThinPassBudget` |
| Proof the base never changes | Device-side fingerprint of frozen tensors before and after every job; a mismatch fails the job | `thin_job.fingerprint`, `thin_job.frozen_sample` |
| Crash recovery | One exclusive lock owns any restart of the pair; re-running recovery on a healthy pair is a no-op | `pair_recovery` |
| Code changes without a restart | Both ranks hash the staged files, agree, then execute exactly the hashed bytes into the existing module objects | `hotload` |

## Measured results (origin setup only)

On the origin setup the best added expert reached:

- **16/24** on the development facts (strict scoring) and **11/24** on reworded questions,
  with the retrieval memory pocket switched **off**, from a baseline of **0/24**.
- The **14 control questions** were unchanged.
- The frozen-weight fingerprint never changed across jobs.

These are small numbers on one private fact set. They show that the mechanism works on
that setup; they do not show general quality.

## Known failures

- **Sibling-value leaks.** Two near-miss facts answered with a value belonging to a
  neighbouring fact. Three approaches to fixing this were tried and **all failed**
  (see `LESSONS.md`). This is unsolved.
- Rewording generalisation is clearly weaker than recall on the trained phrasing (11 vs 16).
- See `LESSONS.md` for the integration bugs found on the way and their fixes.

## Hardware- and model-specific

- Origin: GLM-5.3-Flash (MoE) in EXL3 quantisation, tensor-parallel across **two
  machines** (one rank each), served by a TensorFold-derived engine.
- `src/resident_trainer/engine/` (`resident_ops.py`, `thin_hooks.py`, `bootstrap.py`) is
  tied to that engine's internals and kernels. It is kept as a reference; it is **not
  importable** without that engine and is not covered by tests.
- `thin_job.run`, `thin_native` and `answer` assume four residual streams, a world size of
  two and the origin layer numbers (`RESIDENT_TRAINER_LAYER`, default 35).

## Untested

- Any other model, quantisation, engine, GPU or world size.
- World sizes other than two (several checks hard-require two ranks).
- Long runs beyond the origin experiments.

## Run the CPU tests and toy example

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest                      # CPU only, no GPU or server needed
python examples/toy_train.py
```

The tests cover the streamed loss and layer replay against plain autograd, the paired
start and cancellation protocol with fake decoders and stores, the memory budget and
paired aborts, the recovery lock, hot-load agreement and refusal, and offline recall
scoring. The toy trains added neurons on a frozen 6-layer model and checks that the
frozen fingerprint is unchanged.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RESIDENT_TRAINER_AUTHORITY` | empty (nothing runs) | Exact string a job must carry to run |
| `RESIDENT_TRAINER_RECEIPTS` | `./receipts` | Per-rank receipts, phase logs, stack dumps |
| `RESIDENT_TRAINER_CHECKPOINTS` | `./checkpoints` | Per-rank added-neuron checkpoints |
| `RESIDENT_TRAINER_STATE_DIR` | `./state` | Recovery lock location |
| `RESIDENT_TRAINER_MASTER_PORT` | `29551` | Recorded in the recovery lock |
| `RESIDENT_TRAINER_LAYER` / `_WATCH_LAYER` | `35` / `44` | Trained layer; extra layer included in the frozen hash |

A `STOP` file in the job root aborts a running job at its next boundary.

## Smaller pieces

The coordination, memory, fingerprint, loss, recovery, hot-load and scoring parts also exist as
smaller standalone tools under `tools/`. This folder keeps its own copies so it runs on its own.

## Files

See `AGENTS.md` for a map of the code, the invariants and how to adapt it.
