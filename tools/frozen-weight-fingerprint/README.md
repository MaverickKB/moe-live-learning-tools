# frozen-weight-fingerprint

> Teaching example, not a product. It is a shortcut past a lesson we learned the hard way. Adapt the idea; do not copy it 1:1.

## Problem

Proving that in-place training never changed the base model's weights, cheaply enough to do before and after every job.

## Core idea

- View each tensor's raw bytes as int16 (or uint8).
- Reduce in fixed chunks on the tensor's own device to position-weighted and plain int64 sums.
- Hash the names, shapes, dtypes and sums with SHA-256.
- `frozen_sample` walks chosen layers to collect tensors, including large quantised weights.

## Invariants

- Compare before and after every job; a mismatch fails the job.
- Fingerprint the tensors the trainer could plausibly touch, not just a convenient few.

## Specific to the origin setup

Origin setup: GLM-5.3-Flash (MoE, EXL3) served tensor-parallel across two machines (one rank each) by a TensorFold-derived engine.

- The origin sampled the final norm plus the trained layer and one deeper layer, capped at 96 tensors.

## Adapting it

1. Choose `layers` and `limit` so the sample covers the weights near your training site.
2. For a full proof, fingerprint every tensor once at startup and compare offline.

## Untested or limited

- It is a digest, not a cryptographic hash of the bytes: collisions are very unlikely, not impossible.
- The sample is partial by design; untouched layers outside it are not proven.

## Run the tests

```bash
pip install -e '.[test]'
pytest
```
