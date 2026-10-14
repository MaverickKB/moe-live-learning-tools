# default-drift-probe

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Learning that a new version exists can make a model start recommending it everywhere. This probe catches that.

## Core idea

- Generic prompts where a stable default is expected, plus one case that genuinely needs the new version.
- An answer drifts if it uses a version listed in `avoid_versions` (unless `expects_new`).
- Drift is blamed on the change only when it appears with the change on and not off.

## Invariants

- Always compare on versus off; drift that exists without the change is not its fault.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- The example set uses a public fact (a newest Python release) with six cases.

## Adapting it

1. Write one small case file per awareness fact you add.
2. Plug the on/off answers from your own server into `drift_summary`.

## Untested or limited

- Only one example fact and six cases; never run against a live model.
- Version detection uses the same regex heuristic as the recall scorer.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```
