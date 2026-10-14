# Lessons

Each entry: symptom, cause, fix. Entries marked *inferred* are reconstructed from code and notes, not from a written incident record.

## 1. Capturing all ten layers tripped a swap gate (inferred details)
- **Symptom:** A host memory/swap guard fired during activation capture.
- **Cause:** Captures at all ten candidate layers at once.
- **Fix:** Capture one layer only, and keep the floor check in every capture and train path.
