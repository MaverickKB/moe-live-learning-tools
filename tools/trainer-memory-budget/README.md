# trainer-memory-budget

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Letting a background trainer use spare memory on a serving host without ever pushing the host into swap or out-of-memory, and without stopping serving.

## Core idea

- Admission requires the free-memory floor plus the whole private workspace to be available.
- A monitor thread samples host free memory and device allocation growth; flags record floor, workspace, deadline, authority and sampling errors.
- At every boundary both ranks exchange flag words over a trainer-only channel and abort together if any rank has a flag.
- An abort raises `PassAborted`, which discards the pending update and nothing else.

## Invariants

- The floor aborts only the trainer.
- Aborts happen at the same boundary on every rank.
- Authority is checked before any device work.
- Garbage flag words are a desync, not a flag.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- Default floor 1 GB (decimal) and workspace 2 GiB were tuned for the origin hosts.
- `available_bytes` reads Linux `/proc/meminfo`.

## Adapting it

1. Inject `cuda_bytes` and `read_available` for your device and OS.
2. Supply `exchange` over your side channel.
3. Tune `floor` and `workspace` per host; leave headroom for serving growth.

## Untested or limited

- Non-Linux hosts.
- Hosts with unified memory other than the origin's.
- More than two ranks.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```

See `LESSONS.md` for the failures behind this tool.
