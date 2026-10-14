# recovery-lock

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Several recovery paths (a guard's rollback, a manual restart) racing to restart the same multi-host server.

## Core idea

- Every recovery path takes one exclusive non-blocking file lock.
- The lock file records the owner's pid, purpose and port.
- Recovery is idempotent: if a live guard owns a healthy pair, it does nothing.

## Invariants

- One recovery owner at a time.
- A second attempt fails fast with the first owner's purpose.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- Master port 29551 and the guard-owner JSON format are from the origin setup.

## Adapting it

1. Point `RECOVERY_LOCK_PATH` at storage every recovery path can see.
2. Pass your own `healthy()` and `restore()` callables.

## Untested or limited

- Network filesystems (flock semantics vary).
- Recovery paths on different hosts that do not share the lock file.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```

See `LESSONS.md` for the failures behind this tool.
