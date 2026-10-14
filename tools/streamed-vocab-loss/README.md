# streamed-vocab-loss

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Computing an exact full-vocabulary cross-entropy and its gradient through a frozen stack of layers when memory is too tight for full logits or a full activation graph.

## Core idea

- `StreamedCrossEntropy`: loop over vocabulary tiles with an online log-sum-exp; the backward pass recomputes tiles and streams the input VJP.
- Across tensor-parallel vocabulary shards, partial statistics are combined once.
- `LayerReplay`: keep only layer boundaries in the forward pass; in the backward pass rebuild one layer's graph at a time.
- `accumulated_update`: microbatch 1 accumulation that matches the mean-per-example objective, with global clipping across ranks.

## Invariants

- The base head and layers stay frozen (`TensorHead` rejects a trainable matrix).
- Every target occurs exactly once across shards.
- A guard callback runs between tiles and layers so a budget abort can stop work promptly.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- Four residual streams and the four-stream checkpoint sharding come from the origin model.

## Adapting it

1. Implement a head with `logits(inputs, start, stop)` and `input_vjp(...)` over your frozen classifier.
2. Pass your frozen suffix layers to `replay_suffix`.
3. Remove the four-stream assumptions in `shard_added_checkpoint` if your model has one residual stream.

## Untested or limited

- Mixed precision beyond what the origin used.
- Very long sequences (the origin used short answer spans).

## Run the tests

```bash
pip install -e '.[test]'
pytest
```

See `LESSONS.md` for the failures behind this tool.
