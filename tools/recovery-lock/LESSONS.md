# Lessons

Each entry: symptom, cause, fix. Entries marked *inferred* are reconstructed from code and notes, not from a written incident record.

## 1. Two restarts raced
- **Symptom:** Two restarts bound the same master port (EADDRINUSE).
- **Cause:** A guard's rollback and a direct container restart ran at the same time.
- **Fix:** One lock for every recovery path; re-running on a healthy pair is a no-op.
