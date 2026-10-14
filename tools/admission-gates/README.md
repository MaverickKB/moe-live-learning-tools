# admission-gates

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Deciding which facts may be written permanently into model weights, and which must stay in a retrieval pocket or be dropped.

## Core idea

- Gate 1 (front funnel): classify a candidate by kind and reuse; reject one-off, chatter and personal kinds; admit the rest to the pocket.
- The pocket index tracks state, versions, supersedes, corrections and near-miss siblings.
- Gate 2 (judge): hard rules first (kind, near-miss, volatility, source version, age), then a judge with its own instructions ranks and caps promotions.
- Promotion: consolidate (a consolidation pass, i.e. training or writing into weights), check regressions, roll back on failure, and free a pocket entry only with a passing pocket-off recall receipt.

## Invariants

- Guardrails can only be tightened, never loosened.
- Nothing is freed from the pocket without a pocket-off receipt.
- Near-miss siblings and personal kinds never go into weights under the defaults.
- A corrected fact that was already in weights must be re-trained over the old value.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- The kinds list and judge instructions reflect the origin's use; the judge and classifier were wired to the origin model but never run live.

## Adapting it

1. Edit the kinds and instructions for your domain; keep the locked guardrails.
2. Inject your own classifier, judge, consolidation, recall, regress and rollback callables.

## Untested or limited

- Never run live: everything heavy is a stub in the tests.
- Judge and classifier prompt quality is unmeasured.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```
