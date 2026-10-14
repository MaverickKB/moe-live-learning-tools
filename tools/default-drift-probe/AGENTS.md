# AGENTS.md: default-drift-probe

Use this when you add knowledge about new versions and must keep default recommendations stable.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/drift_probe/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.
- `examples/`: runnable or data examples.

## Must hold

- Always compare on versus off; drift that exists without the change is not its fault.

## To adapt

1. Write one small case file per awareness fact you add.
2. Plug the on/off answers from your own server into `drift_summary`.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
