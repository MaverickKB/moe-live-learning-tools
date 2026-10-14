# Lessons

Each entry: symptom, cause, fix. Entries marked *inferred* are reconstructed from code and notes, not from a written incident record.

## 1. Sibling-value leaks (unsolved)
Two near-miss facts were answered with a value belonging to a neighbouring (sibling) fact.
Three approaches were tried; none fixed it:

1. **Contrast training** with "unknown" examples and cross-pocket examples (sets of 140 and 160
   examples). The leaks remained.
2. **Longer contrast training** (checkpoint c2-step24): 14/24 dev and 8/24 rewordings, down from
   16 and 11, with the leaks still at 2.
3. **Scaling the expert's output gain down** to 0.7 and then 0.5: the leaks stayed at 2 and
   rewordings fell to 10, then 7.

**Conclusion:** the middle-layer expert has no gate, so it cannot decide when *not* to fire.
Exact facts that are easily confused with a sibling belong in retrieval memory, not in the
weights. The admission gates encode this: near-miss siblings never go into weights by default.
