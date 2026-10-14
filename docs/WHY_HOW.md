# Why each tool exists and how it works

One short entry per tool. "Why" points to the lesson that produced it (see
[LESSONS.md](LESSONS.md) and [ORIGIN_STORY.md](ORIGIN_STORY.md)); "How" is the core idea.
Every tool is a starting point to adapt, not a solution. Details are in each tool's README.

| Tool | Why it exists | How it works |
|---|---|---|
| `resident-trainer` | A second model load for training did not fit next to serving, and taking serving down was not acceptable | Trains added shared-expert neurons at one middle layer inside the serving process, between scheduler iterations, reusing loaded weights; base weights frozen |
| `paired-job-start` | Starting a side job on two ranks at different stream points froze serving | The head writes the start into the sealed serving stream; the worker confirms the message number on a separate channel; mismatch or timeout cancels only the job |
| `hot-load` | Restarts were costly and Python reload served stale bytecode | Hash staged files on every rank, agree, then compile and execute exactly those bytes; refuse restart-only files |
| `trainer-memory-budget` | Trainers sharing machines with serving hit swap and allocation limits | Reads free memory on all ranks against a floor; on breach, aborts the trainer on all ranks together, never serving |
| `frozen-weight-fingerprint` | "Base weights frozen" had to be proven, cheaply, on device | Device-side fingerprint of frozen tensors before and after each job; any change fails the job |
| `streamed-vocab-loss` | Full-vocabulary logits did not fit in memory | Computes exact cross-entropy and gradients in vocabulary chunks through the frozen head |
| `recovery-lock` | Concurrent restart paths raced on the same multi-host server | A single-owner lock; other jobs read it and stay out of the way |
| `recall-scorer` | Loss and pocket-on scores overstated learning; sibling values leaked | Scores recall with the pocket off: strict hits, sibling-value leaks, invented specifics |
| `admission-gates` | Exact, easily confused facts leaked when trained into weights | Gate 1 admits into memory; gate 2 (different instructions) picks stable, general facts to train; a pocket entry is freed only on a passing pocket-off receipt |
| `default-drift-probe` | Knowing a thing can sway its use | Probes whether learning about a new version changes default recommendations; drift is blamed only when it appears with the expert on |

All paths, hosts and authorisation strings are settings or environment variables; none are
hard-coded. If you find one, please report it.
