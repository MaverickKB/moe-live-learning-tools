# AGENTS.md: recall-scorer

Use this when you need honest numbers on whether a model learned facts into its weights.

Read `README.md` first. Adapt the idea to your engine; do not copy it 1:1.

## Map

- `src/recall_scorer/`: the implementation.
- `tests/`: CPU tests; fakes show the interfaces you must provide.

## Must hold

- Evaluate with the pocket off, or you are measuring retrieval.
- Keep a sealed held-out set that is never read during development.

## To adapt

1. Bring your own fact set and grader; this module only scores saved answers.
2. Extend the `SPECIFIC` regex for identifiers in your domain.

## Check before claiming it works

- `pytest` passes on CPU.
- Your own engine test reproduces the invariants above on your hardware.
