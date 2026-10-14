# Standalone repository candidates

All tools stay in this collection. These are flagged as possibly worth their own repository
later, with deeper docs and application examples. Nothing here has been built yet.

## 1. resident-trainer: learning inside live serving

- **Why:** this is the main result. A large MoE model learned new facts inside its running
  server, with no downtime and with the original weights provably unchanged. Most published
  approaches fine-tune offline or swap adapters with a reload.
- **Evidence:** on one two-machine setup, from 0/24 to 16/24 dev facts (strict) and 11/24
  rewordings with the retrieval pocket off. 14 controls unchanged. Frozen-weight fingerprint
  unchanged across every job. Chats kept being served between steps.
- **Limits that must be stated in a standalone repo:** one model, one engine, one cluster
  shape, one small private fact set. No independent replication. Sibling-value leaks are
  unsolved. The rewording gap (11 vs 16) shows limited generalisation.
- **Deeper docs needed:** an architecture walkthrough with a timeline of one job across both
  ranks; a porting guide for one common open engine; an evaluation protocol (fact sets,
  pocket-off scoring, controls, held-out set); a failure catalogue with logs.
- **Application examples needed:** a single-GPU or two-process CPU simulation of the full job
  loop; a worked port to one public engine and a small open MoE; a recipe for teaching a codebase's
  conventions and checking controls.
- **Before publishing on its own:** replicate on a second model or engine, publish the eval set
  (or a public equivalent), and report variance across seeds.

## 2. paired-job-start + hot-load: live side jobs for tensor-parallel servers

- **Why:** together they let any side job (training, capture, evaluation, code updates) run
  inside a live multi-rank server without desyncing collectives or restarting. The pattern is
  useful well beyond training.
- **Evidence:** both ran live on the origin setup as part of the resident trainer. They fixed
  three observed failures (serving freeze from collective desync, a backward stall, a post-job
  wait) and the stale-bytecode reload problem. CPU tests reproduce each protocol case with
  fakes.
- **Limits:** only two ranks, only one engine, no pipeline parallelism. On its own this is a
  solid engineering pattern rather than a research result; it is better published as a
  practical guide than called a breakthrough.
- **Deeper docs needed:** sequence diagrams for start, cancel, nested serving and finish; a
  hook-by-hook porting guide; a guide to choosing op codes and timeouts.
- **Application examples needed:** a minimal two-process server (torch.distributed on CPU)
  showing a live job and a hot-load end to end; an example side job that is not training
  (such as a live activation capture).

## Not flagged

trainer-memory-budget, frozen-weight-fingerprint, streamed-vocab-loss, recovery-lock,
recall-scorer, admission-gates and default-drift-probe are useful pieces. They work best as
parts of the collection or as supporting chapters of the repos above. Admission gates and the
drift probe have never run live, so they cannot carry a standalone claim yet.
