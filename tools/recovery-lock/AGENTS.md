# AGENTS.md: recovery-lock

Use this when more than one process may try to restart the same server.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/recovery_lock/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.

## Must hold

- One recovery owner at a time.
- A second attempt fails fast with the first owner's purpose.

## To adapt

1. Point `RECOVERY_LOCK_PATH` at storage every recovery path can see.
2. Pass your own `healthy()` and `restore()` callables.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
