# AGENTS.md: admission-gates

Use this when you need a policy layer deciding what may be trained into weights.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/admission_gates/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.
- `examples/`: runnable or data examples.

## Must hold

- Guardrails can only be tightened, never loosened.
- Nothing is freed from the pocket without a pocket-off receipt.
- Near-miss siblings and personal kinds never go into weights under the defaults.
- A corrected fact that was already in weights must be re-trained over the old value.

## To adapt

1. Edit the kinds and instructions for your domain; keep the locked guardrails.
2. Inject your own classifier, judge, dream, recall, regress and rollback callables.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
