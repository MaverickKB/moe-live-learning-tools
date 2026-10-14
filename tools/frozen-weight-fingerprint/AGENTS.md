# AGENTS.md: frozen-weight-fingerprint

Use this when you train next to frozen weights and want evidence that they did not change.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/frozen_fingerprint/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.

## Must hold

- Compare before and after every job; a mismatch fails the job.
- Fingerprint the tensors the trainer could plausibly touch, not just a convenient few.

## To adapt

1. Choose `layers` and `limit` so the sample covers the weights near your training site.
2. For a full proof, fingerprint every tensor once at startup and compare offline.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
