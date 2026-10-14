# Research lessons: what worked, what failed, and why

This is the project-wide lesson log behind the tools in this collection. Per-tool incident
lessons stay next to each tool (`tools/*/LESSONS.md`); this file covers the research path.

**Ground rules for this file.** Every number below was copied from a recorded result. Where a
lesson is remembered but we could not find a written measurement, it says so. One model
(GLM-5.3-Flash, EXL3, tensor-parallel across two machines), one engine (TensorFold-derived),
small private fact sets. Nothing here has been independently replicated. Treat every lesson
as a starting point for your own measurement, not as a law.

The chronological story is in [ORIGIN_STORY.md](ORIGIN_STORY.md).

---

## What worked

### 1. Attention-readable memory slots, trained against the model's own docs-in-prompt behaviour
- **What:** a small set of learned key/value rows ("slots") that the frozen model's attention
  reads, fitted so the model with slots and *no facts in the prompt* behaves like the same model
  with the facts in the prompt (the teacher).
- **Measured:**
  - CPU prototype (Qwen2.5-0.5B-Instruct, frozen): learned 32-row cartridge **24/24** held-out
    answers grounded, baseline 0/24, facts-as-text 22/24, full numerical cache 22/24.
  - Native on GLM, 32 rows: **16/24** learned vs 24/24 complete memory and 1/24 baseline.
    Interpretation recorded at the time: "carrier works; compression misses identifiers".
  - Native, 64 rows (fresh set): 27/36 grounded, 30/30 controls exact, zero off-path memory
    writes. Rejected as not reliable enough for six facts.
  - Native, 96 rows at eleven full-attention sites: 34/36 grounded vs 1/36 baseline and 35/36
    complete memory, 30 exact unrelated and near-miss controls. Accepted.
  - A later precision continuation (224 rows, sealed fresh set): 24/24 grounded, 2/24 baseline,
    30 exact tagged controls.
- **Why it worked (our reading, not proven):** the teacher is the model itself, so the target is
  something the model already knows how to do (read facts from context). Fitting used real
  gradients against attention-reference outputs on frozen trajectories.
- **Limits:** slot files are small, but the full native anchor that went with them was about
  74.6 MB, dominated by recurrent state. Catalogue scale and many combined pockets are unproven.

### 2. Real-gradient training of a small added block, distilled from docs-in-prompt answers
- Added shared-expert neurons at a **middle** layer, base weights frozen, trained with real
  backward passes. Best checkpoint: 0/24 to **16/24** dev strict and **11/24** rewordings with
  the pocket off, 14/14 controls unchanged, 2 sibling-value leaks. See
  [live-serving-expert-training](https://github.com/MaverickKB/live-serving-expert-training).

### 3. Training inside live serving, base weights proven frozen
- Training ran inside the serving process, reusing loaded weights; chats were served between
  microbatches. A device-side fingerprint of the frozen weights was taken before and after
  every job and never changed. Tools: `resident-trainer`, `frozen-weight-fingerprint`.

### 4. Paired sealed starts and hot-load of exact hashed bytes on a TP=2 pair
- Side jobs on two ranks must start at the same point of the serving stream (the first
  attempt without this froze serving for everyone). Code changes are hashed on both ranks,
  agreed, then exactly those bytes are executed. Tools: `paired-job-start`, `hot-load`.

### 5. Measurement discipline
- **Pocket-off recall scoring:** consolidation only counts if the fact is recalled with the
  memory switched off. Several "wins" disappeared under this rule.
- **Judged controls:** exact-text control matching flagged harmless rewordings as regressions,
  so control changes are now judged (a change counts as a drop only if graded worse; an
  unparseable judgement counts as worse). Recorded in the build notes; the judge is unmeasured
  against a labelled set.
- **Default drift:** knowing a new version must not change default recommendations
  (`default-drift-probe`). Built, never run live.
- **Fresh, sealed held-out sets**, opened once. A frozen eval file was once found with two
  answers swapped; it was fixed and re-sealed *before* any scored run. Check your eval data.

---

## What failed, and why we think it failed

| Approach | Recorded result | Why (our best reading) |
|---|---|---|
| **Last-layer neuron pockets** (64 neurons at the final MoE layer) | Coding lesson: 3/20 substance (best version) vs 0/20 baseline and 16/20 docs-in-prompt. Simple facts: 7/24 (5/24 fully grounded) vs 24/24 docs-in-prompt; pocket-off 0/24 | The last layer can only nudge the next token; it cannot re-route the computation earlier layers already did. Offline token recall looked good (up to ~95% on dev) but did not turn into generated answers. |
| **Added width alone** | Same last-layer setup at 64/128/256 neurons: dev fact recall 27.9% / 22.1% / 20.9% | More neurons in the wrong place did not help; the measured knowledge-delta span was low-rank (95% variance in 58 directions over 446 rows), so the limit was placement and objective, not capacity. |
| **Gradient-free (SPSA / derivative-free) fitting** | Shared-expert extension over ten layers with SPSA output fit: 0/20 (baseline 0/20, docs 8/20), and one protected answer changed. MEMIT campaign SPSA screens: best 1/12, and that one hit invented details | A few dozen random-direction coefficients through a quantised multi-layer stack is too weak a signal; objective went from 1.304 to 1.263 and generated recall did not move. |
| **MEMIT-style multi-layer edits** | Six closed-form passes (raw-cosine and regularised keys, three seeds each): 0/12 dev. Stopped under the three-strikes rule | Closed-form key/value edits assume a clean linear store; on this hybrid quantised model the keys did not separate personal facts well enough, and editing base weights also conflicts with the "base weights frozen" goal. |
| **Gap-picked targets** | *Partly verified.* The brief that followed lists "gap-picked targets" among failures. The written record shows trigger layers and rows chosen by separation/margin (the Living Weights recipe ranks layers by silence on negatives, coverage, then widest margin); the ten-layer run selected that way scored 0/20 | Picking where the activation gap is widest optimises for a clean trigger, not for whether the output changes the answer. We could not find a write-up isolating this as a single variable. |
| **Neuron consolidation from the memory teacher, approximate backward** | 0/12 dev on four screens, controls failed; rejected | Native forward matched, but the straight-through backward had measured derivative disagreement, so training followed a wrong gradient. |
| **Closed-form KV-pocket to FFN mapping (CPU research)** | Preserved fitted controls but failed reliable unseen recall | Fitting the seen rows is easy; generalising to rewordings is not. |
| **Sibling-value fixes** (contrast data, longer training, output gain 0.7/0.5) | Leaks stayed at 2 in every attempt; rewordings fell to 8, 10, 7 | The added block has no gate, so it cannot decide when not to fire. Easily confused exact facts belong in memory, not weights. |

---

## Operational lessons

### Memory floor and trainer abort
A trainer sharing machines with live serving must watch free memory and abort **only itself,
on all ranks together**. Recorded examples: a learning-rate sweep was aborted three times at
start-up by a swap-out gate on one host (the floor itself held) and the same sweep then ran
cleanly on the other host, where every learning rate lowered loss (best: 1.035 to 0.528 at
lr 0.01; a matched output-head-only control only reached 1.030). A later native fit failed its
first CUDA allocation at about 2.6 GB driver headroom; the next attempt added an explicit
allocator headroom check. Floors used over time: 4 GB, later 1 GB free per machine with a
2 GiB trainer ceiling. Tool: `trainer-memory-budget`.

### Side jobs must not kill their neighbours
Other agents and services shared the same server port and machines. Rules that grew out of
that: announce downtime windows in a shared contract file, read (never take) a recovery lock
someone else holds, stop a trainer when a live user needs the server (recorded: a training
run was stopped for a live user and serving restored), and keep a single owner for pair
recovery. Tool: `recovery-lock`. *We did not find a written incident where a side job killed
a neighbour process; these rules are preventive.*

### The request-extent bug
Under concurrent load, installing memory for a request raised "request extent too small for
memory". The two ranks then fell out of step, so real requests failed with HTTP 500 **while
the health check still returned 200**. Fix: reserve the effective memory prompt plus its
decode window at admission, agree preparation on both ranks before the forward, release only
the failed request, and make health return 503 on rank desync. Verified with 56 CPU tests,
concurrent mixed load and long prompts. Lesson: a health check that cannot see rank desync is
not a health check.

### Three-strikes rule
If the same failure repeats three times, stop and change the approach rather than tuning.
Recorded uses: a compressed-memory fit kept missing the same identifier on three screens
(stopped, carrier changed); the MEMIT campaign was stopped under it; GPU command-buffer hangs
on one local backend led to a switch to CPU. The rule is now built into code (a fit stops when
the same dev miss repeats three times; the outer loop stops after three identical rounds).

### Briefs that were too tight
*Not verified from a written record.* We recall that several agent jobs failed or stalled
because their briefs fixed the method too narrowly (for example, prescribing a fitting recipe
or forbidding the change that the evidence pointed to). We could not locate a write-up that
measures this, so treat it as an unverified working lesson: state the goal, the hard safety
lines and the success bar; leave the method open; allow the agent to change approach after
three strikes.

### Hard-coded paths, values and the scrub
Research code accumulated hard-coded lab paths, hosts and operator strings (for example an
audit script tied to one experiment directory, and an authorisation string baked into a
launcher). Before publishing, everything was moved to environment variables or settings with
placeholder defaults, and the public trees were scanned for personal data, hosts, addresses
and home paths (plus a secrets scanner). We publish this as a lesson rather than hide it: make
every location a setting from day one, and run a scrub before every push.

### Honest labels
Several results were first reported more favourably than the evidence allowed (for example a
self-graded 18/18 whose "controls" included inferable facts; an intermediate tally that missed
later passes). Each was corrected in writing. Label what is measured, what is CPU-only, and
what is untested.
