# AGENTS.md: trainer-memory-budget

Use this when a background job shares a host with serving and must back off before memory runs out.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/trainer_budget/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.

## Must hold

- The floor aborts only the trainer.
- Aborts happen at the same boundary on every rank.
- Authority is checked before any device work.
- Garbage flag words are a desync, not a flag.

## To adapt

1. Inject `cuda_bytes` and `read_available` for your device and OS.
2. Supply `exchange` over your side channel.
3. Tune `floor` and `workspace` per host; leave headroom for serving growth.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
