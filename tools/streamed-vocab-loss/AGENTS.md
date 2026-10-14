# AGENTS.md: streamed-vocab-loss

Use this when you need exact loss and gradients through a frozen model under a tight memory budget.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/streamed_loss/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.
- `examples/`: runnable or data examples.

## Must hold

- The base head and layers stay frozen (`TensorHead` rejects a trainable matrix).
- Every target occurs exactly once across shards.
- A guard callback runs between tiles and layers so a budget abort can stop work promptly.

## To adapt

1. Implement a head with `logits(inputs, start, stop)` and `input_vjp(...)` over your frozen classifier.
2. Pass your frozen suffix layers to `replay_suffix`.
3. Remove the four-stream assumptions in `shard_added_checkpoint` if your model has one residual stream.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
