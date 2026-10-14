# moe-live-learning-tools

Teaching examples from a research project on teaching a large mixture-of-experts model new
facts **while it keeps serving**. They are the shortcuts we wish we had had: each tool
captures one hard lesson. They are **not products** and are not meant to be copied 1:1.
Adapt the ideas to your engine, model and cluster.

> **About this work.** This is an honest, ongoing investigation into what is possible, not a
> claim of perfection. We make mistakes and backfill transparency as we go, in the hope that
> others benefit and can skip the hard lessons we hit.

## The two results that matter

1. **Learning inside live serving, without taking the model down.** A small set of added
   shared-expert neurons was trained at a middle layer of GLM-5.3-Flash (EXL3, tensor-parallel
   across two machines) inside the running server, reusing the weights it already had loaded.
   Chats kept being served between training steps.
2. **The original weights were never modified.** Only the added neurons were trained. A
   device-side fingerprint of the frozen weights was taken before and after every job, and it
   never changed.

Measured on that one setup: the best added expert went from 0/24 to **16/24** on development
facts (strict) and **11/24** on reworded questions, with the retrieval pocket switched off.
The 14 control questions were unchanged. Two near-miss facts leaked a neighbouring fact's
value, and three attempted fixes failed; exact, easily confused facts belong in memory,
not weights. Nothing here has been run on other hardware.

## Tools

| Tool | Problem it solves | Status |
|---|---|---|
| [`resident-trainer`](tools/resident-trainer/) | Train added shared-expert neurons inside a live tensor-parallel MoE server without downtime and without changing base weights | Ran live on the origin setup; CPU tests pass; engine glue reference-only |
| [`paired-job-start`](tools/paired-job-start/) | Start, pause and finish a side job in lockstep on two TP ranks inside live serving | Protocol ran live inside the resident trainer; extracted version CPU-tested only |
| [`hot-load`](tools/hot-load/) | Change running code on all ranks without a restart, executing exactly the hashed bytes | Used live on the origin setup; extracted version CPU-tested only |
| [`trainer-memory-budget`](tools/trainer-memory-budget/) | Keep a background trainer above a free-memory floor; abort only the trainer, on all ranks together | Ran live inside the resident trainer; extracted version CPU-tested only |
| [`frozen-weight-fingerprint`](tools/frozen-weight-fingerprint/) | Prove frozen base weights never changed, cheaply, on device | Ran live before/after every job on the origin setup; CPU-tested |
| [`streamed-vocab-loss`](tools/streamed-vocab-loss/) | Exact full-vocabulary loss and gradients through a frozen stack under tight memory | Ran live inside the resident trainer; CPU parity tests |
| [`recovery-lock`](tools/recovery-lock/) | Stop concurrent restart paths racing on the same multi-host server | Used on the origin setup; CPU-tested |
| [`recall-scorer`](tools/recall-scorer/) | Score pocket-off recall honestly: strict hits, sibling-value leaks, invented specifics | Origin results were measured with an earlier, infrastructure-bound scorer; this engine-free version is CPU-tested only |
| [`admission-gates`](tools/admission-gates/) | Decide which facts may be trained into weights and when a pocket entry may be freed | Built and CPU-tested with stubs; never run live |
| [`default-drift-probe`](tools/default-drift-probe/) | Check that learning about a new version does not shift default recommendations | Built and CPU-tested; never run against a live model |

`tools.json` is the same catalog in machine-readable form. `docs/STANDALONE_CANDIDATES.md`
lists the tools that may deserve their own repository later.

## How these chain together (direction, not a finished system)

```
new fact
  -> admission-gates (gate 1)    admit into the retrieval pocket, or reject
  -> pocket                      answers come from retrieval meanwhile
  -> admission-gates (gate 2)    judge picks stable, general facts to consolidate
  -> resident-trainer            train added neurons inside live serving
       uses paired-job-start, trainer-memory-budget, streamed-vocab-loss,
            frozen-weight-fingerprint, hot-load, recovery-lock
  -> recall-scorer               measure recall with the pocket OFF
  -> default-drift-probe         + control and general-chat regression checks
  -> free the pocket entry only on a passing pocket-off receipt, else roll back
  -> repeat
```

The training step and the measurement steps ran live on the origin setup. The admission
gates, the automated loop around them and the drift probe were built and CPU-tested but have
not run end to end. This is the direction being refined.

## Standalone repositories

Two of these tools also have their own repositories with deeper docs and examples:

| Repository | Covers | URL |
|---|---|---|
| live-serving-expert-training | resident-trainer: architecture, porting guide, evaluation protocol, results | `https://example.invalid/TODO-fill-at-publish/live-serving-expert-training` (placeholder) |
| tensor-parallel-side-jobs | paired-job-start + hot-load: sequence diagrams, porting guide, two-process CPU example | `https://example.invalid/TODO-fill-at-publish/tensor-parallel-side-jobs` (placeholder) |

**TODO at publish time:** replace both placeholder URLs with the real repository URLs.

## Running tests

Each tool is self-contained:

```bash
cd tools/<tool>
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest
```

All tests run on CPU without GPUs, servers or network.

## License

MIT for the original code, with Apache-2.0 portions derived from upstream projects. See
`LICENSE`, `NOTICE` and `licenses/`.
