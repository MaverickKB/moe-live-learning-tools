# Lessons

Each entry: symptom, cause, fix. Entries marked *inferred* are reconstructed from code and notes, not from a written incident record.

## 1. Learning-rate scale built through a probe (inferred details)
- **Symptom:** Effective learning rates did not match the requested scale.
- **Cause:** Optimizer state was first created on the probe path, so the stored base rates were wrong.
- **Fix:** Create optimizer state only from the checkpoint, record the original rates once, and scale from those.
