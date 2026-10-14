# engine/ (reference only)

These files are tied to the origin serving engine (a TensorFold-derived GLM-5.3-Flash EXL3
stack) and are kept so that you can see how the trainer was wired in:

- `resident_ops.py`: native forwards and frozen input VJPs over the engine's kernels.
- `thin_hooks.py`: registration of trainer callbacks into the engine's request hooks.
- `bootstrap.py`: source overlay that installs the hooks at engine import time.

They import modules (`tensorfold`, `tensorfold_host`) that are **not included**. They are
not importable here and are not covered by tests. Port them to your engine rather than
running them.
