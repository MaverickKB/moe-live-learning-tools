# AGENTS.md: hot-load

Use this when you must change running Python code on several ranks at once without a restart.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/hot_load/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.

## Must hold

- All ranks load or none do.
- Only bytes that were hashed are executed (no bytecode cache, no re-read after verify).
- Files whose objects are held by live hooks or closures are restart-only.

## To adapt

1. Pass your own `order` (leaves first, the module whose frame is running last) and `restart_only`.
2. Provide `exchange(name, value) -> [value per rank]` over any side channel both ranks share.
3. Run the load at a point where both ranks are in the same state (a paired job).

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
