# Lessons

Each entry: symptom, cause, fix. Entries marked *inferred* are reconstructed from code and notes, not from a written incident record.

## 1. Stale bytecode on reload
- **Symptom:** After a code change a rank kept running old code.
- **Cause:** Python's normal reload path can serve cached bytecode that looks fresh (same size and mtime).
- **Fix:** Hash on every rank, agree, then compile and exec exactly those bytes. `test_executes_new_bytes_even_when_size_and_mtime_match` reproduces the trap.
