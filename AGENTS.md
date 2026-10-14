# AGENTS.md (index)

**Adapt the ideas; do not copy these tools 1:1.** They are teaching examples from one
two-machine setup. Each tool's README says what is specific to that setup and what is untested.

| Tool | When to use it |
|---|---|
| [resident-trainer](tools/resident-trainer/AGENTS.md) | Use this when you want the full worked example of training inside a live tensor-parallel server. |
| [paired-job-start](tools/paired-job-start/AGENTS.md) | Use this when you need two tensor-parallel ranks to start, pause and finish a side job in lockstep inside live serving. |
| [hot-load](tools/hot-load/AGENTS.md) | Use this when you must change running Python code on several ranks at once without a restart. |
| [trainer-memory-budget](tools/trainer-memory-budget/AGENTS.md) | Use this when a background job shares a host with serving and must back off before memory runs out. |
| [frozen-weight-fingerprint](tools/frozen-weight-fingerprint/AGENTS.md) | Use this when you train next to frozen weights and want evidence that they did not change. |
| [streamed-vocab-loss](tools/streamed-vocab-loss/AGENTS.md) | Use this when you need exact loss and gradients through a frozen model under a tight memory budget. |
| [recovery-lock](tools/recovery-lock/AGENTS.md) | Use this when more than one process may try to restart the same server. |
| [recall-scorer](tools/recall-scorer/AGENTS.md) | Use this when you need honest numbers on whether a model learned facts into its weights. |
| [admission-gates](tools/admission-gates/AGENTS.md) | Use this when you need a policy layer deciding what may be trained into weights. |
| [default-drift-probe](tools/default-drift-probe/AGENTS.md) | Use this when you add knowledge about new versions and must keep default recommendations stable. |

Machine-readable catalog: `tools.json` (name, path, problem, status, tested_on, deps,
standalone_candidate).

## Rules for every tool
- Each tool is self-contained (`pyproject.toml`, `src/`, `tests/`, `README.md`, `AGENTS.md`,
  and `LESSONS.md` where real lessons exist). Some small code is duplicated on purpose so each
  tool stands alone; there is no shared package. Add `common/` only for code two tools truly
  share at runtime.
- Run the tool's `pytest` before and after a change. Tests must pass on CPU.
- No user-specific paths, hosts, IPs or usernames. Use environment variables with generic
  defaults and placeholder fixtures (`host-a.example`, `atlas`, documentation IP ranges).
- Claims stay within what was measured. Mark everything else untested, and mark lessons whose
  mechanism was reconstructed as *inferred*.
