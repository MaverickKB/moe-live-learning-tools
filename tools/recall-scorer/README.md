# recall-scorer

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Measuring whether facts live in the weights, not in retrieval, and catching the ways recall goes wrong.

## Core idea

- Score answers produced with the retrieval pocket switched off.
- Strict credit needs a grader pass and the expected value in the text (whole-token match).
- Sibling leaks: a failed answer that names another fact's value.
- Invented specifics: identifiers (versions, hosts, paths, flags, IPs) in the answer that appear in neither the facts nor the question.
- `drop_pct`: share of paired cases that got worse with the change on.

## Invariants

- Evaluate with the pocket off, or you are measuring retrieval.
- Keep a sealed held-out set that is never read during development.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- The origin used 24 development facts, 24 rewordings, near-miss pairs and 14 controls.

## Adapting it

1. Bring your own fact set and grader; this module only scores saved answers.
2. Extend the `SPECIFIC` regex for identifiers in your domain.

## Untested or limited

- Invented specifics is a regex heuristic; plain-word inventions are missed.
- Leak detection only knows the values you pass in.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```

See `LESSONS.md` for the failures behind this tool.
