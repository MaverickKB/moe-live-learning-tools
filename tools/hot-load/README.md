# hot-load

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Changing stage code in a running multi-rank server without a restart, and being sure every rank runs exactly the same new code.

## Core idea

- Each rank hashes the staged files and refuses anything on a restart-only list.
- Ranks exchange verdicts; any mismatch or problem refuses the load on every rank.
- Each rank compiles and `exec`s exactly the hashed bytes into the existing module object, so callers that look names up at call time see new code.
- Ranks exchange the list of reloaded modules and fail if it differs.

## Invariants

- All ranks load or none do.
- Only bytes that were hashed are executed (no bytecode cache, no re-read after verify).
- Files whose objects are held by live hooks or closures are restart-only.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- The default `ORDER` and `RESTART_ONLY` lists name the origin trainer's modules.
- `exchange` was the trainer-only channel inside a paired sealed job (see `paired-job-start`).

## Adapting it

1. Pass your own `order` (leaves first, the module whose frame is running last) and `restart_only`.
2. Provide `exchange(name, value) -> [value per rank]` over any side channel both ranks share.
3. Run the load at a point where both ranks are in the same state (a paired job).

## Untested or limited

- Closures already installed keep their old bodies until a restart; only names looked up at call time change.
- Module-level state is re-executed: globals are reset unless the module guards them.
- Only ever ran with two ranks.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```

See `LESSONS.md` for the failures behind this tool.
