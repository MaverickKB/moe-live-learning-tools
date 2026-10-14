# Origin story: how these tools came to exist

A documentary-style, traceable account of the research behind this collection and its two
sibling repositories (live-serving-expert-training, tensor-parallel-side-jobs). Each step
records **what we believed**, **the evidence or research**, **the decision**, and **the
result**, so a reader can follow how the thinking evolved, including the dead ends.

**How to read this.** Dates come from recorded job notes and result-file times (US Eastern),
October 2026; where a time is only a file time it is marked *approx.* Numbers are copied from
written results. Items we remember but could not find written down are marked *unverified*.
This is one team's path on one model and one two-machine tensor-parallel setup, not a recipe.
Lessons in condensed form: [LESSONS.md](LESSONS.md).

**Credit.** The serving recipe and engine we started from are upstream work (TensorFold and
the Mia stack; see `NOTICE` and `licenses/`). The "Living Weights" design in TensorFold v1.0.7
(shared-expert neuron pockets with learned triggers, documented for a Nemotron model) shaped
much of the early thinking.

---

## Step 0. The goal

- **Belief:** a served mixture-of-experts model should be able to learn new facts while it
  keeps serving, and later recall them without the facts in the prompt.
- **Constraints we set ourselves:** never take the two-machine tensor-parallel pair apart,
  keep base weights frozen and provably so, always be able to restore the exact original
  serving pair, and score honestly on held-out questions.

## Step 1. One gradient step on a saved lesson (Oct 10 to 11, approx.)

- **Belief:** a small low-rank adapter trained from a captured lesson would show the signal.
- **Evidence:** a learning-rate sweep on one host was aborted three times at start-up by a
  swap-out gate (the 4 GB floor held). Rerun on the other host: every learning rate lowered
  loss (1.035 to 0.528 at lr 0.01), while a capacity-matched output-head-only control barely
  moved (1.035 to 1.030).
- **Decision:** treat memory gates as first-class; move training to wherever headroom is.
- **Result:** native descent evidence for one saved lesson, but loss is not recall. This is
  where `trainer-memory-budget` and the "score generated answers, not loss" habit began.

## Step 2. Living-Weights-style neuron pockets (Oct 11)

- **Belief:** following TensorFold Living Weights, extra neurons appended to the always-on
  shared expert, with gates keyed to the fact, could store facts.
- **Research:** Living Weights v1.0.7 (64-neuron pocket per shared expert, about one neuron
  per fact per layer across ~10 layers, layers ranked by silence on negatives, coverage, then
  widest margin). GLM-5.3-Flash was checked to have one shared expert per MoE layer.
- **Decision and results, in order:**
  1. *CPU pocket math* (09:05 approx.): 13 tests pass, toy recall 0/2 to 2/2, protected logits
     unchanged. Real fit blocked: saved captures lacked the needed inputs.
  2. *Ten-layer shared-expert extension, margin-selected layers, derivative-free (SPSA)
     output fit* (12:41 approx.): **0/20** on a coding lesson (baseline 0/20, docs-in-prompt
     8/20); one protected answer lost its code fence. Rejected.
  3. *Gradient pocket at the last layer (layer 44)* (15:35 approx.): offline dev token recall
     rose to ~95%, but generated substance only **3/20** (docs-in-prompt 16/20). Width
     64/128/256 did not help (fact recall 27.9/22.1/20.9%).
  4. *Simple personal facts, last layer* (16:11 to 16:44 approx.): **7/24** (5/24 fully
     grounded) vs docs-in-prompt 24/24; pocket-off 0/24; 20/40/64 active neurons no better.
- **What we learned:** the docs-in-prompt arm was always far ahead. The model already knows
  how to use facts *it can read*; the last layer is too late to change the answer, and
  derivative-free fitting is too weak a signal. Added width alone does not fix placement.

## Step 3. Direct weight editing, MEMIT-style (Oct 11, evening)

- **Belief:** closed-form multi-layer key/value edits might succeed where pockets failed.
- **Evidence:** six closed-form passes (two key variants, three seeds) scored 0/12 dev; SPSA
  refinement best 1/12, and that hit invented hardware details.
- **Decision:** stopped under the **three-strikes rule**; exact original pair restored and 8
  protected replies verified byte-exact.
- **Result:** abandoned edits to base weights; reinforced "base frozen, add capacity".

## Step 4. The turn: memory the model can *read* (Oct 11 night to Oct 12)

- **Belief:** since docs-in-prompt kept winning, give the model learned rows that its own
  attention reads, trained to reproduce its own docs-in-prompt behaviour.
- **Research:** a literature sweep (22:53 approx.) re-read Living Weights from source and
  surveyed memory-token and KV-cache approaches.
- **Results:**
  - CPU prototype on a small open model (23:33 approx.): learned 32-row cartridge **24/24**
    held-out vs 0/24 baseline and 22/24 facts-in-text.
  - Native 32 rows on GLM: **16/24** vs 24/24 complete memory ("compression misses
    identifiers"). The local GPU backend hit command-buffer hangs, so we switched to CPU
    rather than trust the loss; a repeated identical dev miss triggered three strikes.
  - Native 64 rows (06:48 approx., Oct 12): 27/36, controls 30/30 exact. Rejected.
  - Native 96 rows at eleven full-attention sites: **34/36** vs 35/36 complete memory.
    **Accepted.** A precision continuation (14:42 approx.) reached 24/24 on a sealed set.
- **Honesty correction:** a first self-graded 18/18 cycle used "controls" that included
  inferable facts. A fresh confirmation with blind scope review and sealed hashes was run
  (18/18 learned, 0/18 baseline, 30 exact scope-valid controls).
- **Decision:** attention memory ("pockets") becomes the staging store and the teacher.

## Step 5. Serving the pockets under real load (Oct 12 to 13)

- **Evidence:** under concurrent load, memory install raised "request extent too small";
  the ranks desynchronised, requests got HTTP 500 while health said 200.
- **Decision:** fix admission sizing, agree on both ranks before forward, fail one request
  only, report desync in health.
- **Result:** fixed and verified under concurrent and long-prompt load; patch released in the
  public glm-rsi-dream repository.

## Step 6. Research round before the dream (Oct 13 morning)

Three parallel research jobs (11:46 to 11:57 approx.): a literature survey of continual
knowledge injection and memory-to-weight consolidation; a closed-form pocket-to-FFN mapping
(preserved controls, failed reliable unseen recall); and a design for a small trainable
expert trained with RL. Sources cited in the dream design included context distillation,
on-policy context distillation, FFN-as-key-value-memory, LoRA, Doc-to-LoRA, MEMIT and its
sequential-edit limits, and Living Weights.

## Step 7. The dream: consolidate into one shared-expert pocket, inside live serving

- **Belief:** distil the pocket (teacher) into exactly **one** added block on the always-on
  shared expert at a **middle** layer, with real gradients and base weights frozen. No new
  routed experts, to keep MoE efficiency.
- **Evidence along the way:** an earlier consolidation with an approximate backward failed
  (0/12 dev; measured derivative disagreement). Separate training processes needed a second
  model load the machines could not afford.
- **Decision:** train **inside** the serving process, reusing loaded weights: paired sealed
  starts on both ranks, nested serving between microbatches, streamed full-vocabulary loss,
  a free-memory floor (1 GB per machine) with trainer-only abort, and a frozen-weight
  fingerprint before and after every job.
- **Problems hit and fixed:** collectives desync froze serving (fixed with the sealed start);
  a half-read scheduler iteration stalled backward (explicit resume op); a post-job wait
  delayed a live chat (doorbell skip); stale bytecode on reload (hashed hot-load).
- **Result:** best expert at layer 35: **0/24 to 16/24** dev strict, **11/24** rewordings,
  pocket off, 14/14 controls unchanged, frozen fingerprint unchanged. **Two sibling-value
  leaks** survived three fixes (contrast data, longer training, gain scaling).
- **Decision:** exact, easily confused facts stay in memory; only stable, general knowledge
  is eligible for weights. This became the two-gate admission design.

## Step 8. The loop and the scorers (Oct 13, CPU-built)

Two-tier admission, pocket-off recall scoring, judged controls (exact matching flagged
harmless rewordings), invented-specifics and default-drift checks, a hot-load path so later
code changes need no restart, and an outer loop that stops on the pass bar, the memory
floor, three identical rounds, or a time cutoff. **Built and CPU-tested only; never run
end to end live.**

## Step 9. Extracting the tools (Oct 13)

- **Decision:** publish the reusable pieces as teaching tools with their incident lessons,
  not as a product. Before publishing, hard-coded lab paths, hosts and operator strings were
  moved to settings, and the trees were scrubbed for personal data.
- **Unverified:** we recall that some agent jobs failed because their briefs fixed the method
  too tightly; we found no written measurement of this.

## Where it stands

Learning inside live serving with frozen base weights works on one setup with a partial
score (16/24, 11/24 reworded). Sibling leaks are unsolved. The automated loop is untested
live. Nothing has been replicated elsewhere. Reports of failures are as welcome as successes.
