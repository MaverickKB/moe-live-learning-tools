# Lessons learned

Each entry: what failed, why, and what fixed it. All were found on the origin two-machine
setup during live serving. Entries marked *(inferred details)* have a confirmed one-line
summary, but the mechanism described was reconstructed from the code, not from an incident write-up.

## 1. Collective desync froze serving
- **Symptom:** serving hung for every user when a trainer job started.
- **Cause:** the two ranks entered the trainer's collectives at different points of the
  serving stream, so a trainer all-gather paired with a serving all-gather.
- **Fix:** the paired sealed start. The head writes `THIN_START` into the sealed serving
  sequence; the worker sees it at the same message number and confirms that number on a
  trainer-only channel. No match or no join within the timeout cancels the job only.

## 2. Backward-pass stall from a half-read iteration
- **Symptom:** the backward pass stalled while chats were waiting.
- **Cause:** one head scheduler iteration can flush several sealed messages. The worker
  read only one of two, then returned to the job while the head was still serving.
- **Fix:** the head ends every nested serving iteration with an explicit `THIN_RESUME`
  op; the worker keeps following messages until it sees it.

## 3. Post-job wait while a chat was live
- **Symptom:** after a job, the worker blocked waiting for a doorbell while a chat was
  live, delaying that chat.
- **Cause:** the head rings the doorbell only when idle; after a resume that left chats
  live, it was not idle and never rang.
- **Fix:** the doorbell skip. If the pair is not idle after the job, the worker returns to
  reading the next message instead of waiting for the bell.

## 4. Capturing all ten layers tripped a swap gate (inferred details)
- **Symptom:** a host-level swap/memory guard fired during capture.
- **Cause:** capturing activations at all ten candidate layers at once.
- **Fix:** capture one layer only (the trained layer). Combined with the free-memory floor.

## 5. Learning-rate scale bug (inferred details)
- **Symptom:** effective learning rates differed from the requested `lr_scale`.
- **Cause:** optimizer state was first built through the probe path, so the stored base
  learning rates were not the checkpoint's.
- **Fix:** build optimizer state only from the checkpoint and record the original rates
  (`orig_lrs`) once; scale from those.

## 6. Capture race and missing hook registration (inferred details)
- **Symptom:** occasional empty or mismatched captures; some jobs saw no hook at all.
- **Cause:** a capture could be read before it was complete, and a hook was not
  registered on one path.
- **Fix:** complete pending captures explicitly before use and write a registration
  receipt per rank so a missing hook is visible before any job runs.

## 7. Stale bytecode on reload
- **Symptom:** after a code change, a rank kept running the old code.
- **Cause:** Python's normal reload path can serve stale cached bytecode.
- **Fix:** `hotload` hashes the staged file on both ranks, agrees, then compiles and
  executes exactly those bytes into the existing module object.

## Unsolved: sibling-value leaks
Two near-miss facts were answered with a value belonging to a neighbouring fact. Three
approaches were tried (not recorded in detail here) and none fixed it without hurting recall or controls. Use
`recall_scoring.sibling_leaks` to measure this on your own fact set. Treat it as open.
