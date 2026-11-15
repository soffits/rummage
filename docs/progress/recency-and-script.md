# Recency term and script-family fairness in ranking

Goal: newer items first among plausible matches (search and recall), and cross-language queries decided by meaning
(queue: "Cross-language miss", "Recall ignores how recent a lead is", items 9 and 10).

## Findings so far
- Cross-language miss is mostly the meaning score, not the word bonus: the embedding scores same-script passages
  higher for the same query, so z is centred per (modality, script family), like the per-modality centring.
- Word bonus: scaled by the share of meaning candidates in the query's script family (the word channel can't see
  the rest); lookups (IDs) keep full weight.
- Recency: additive bonus, exponential decay with a half-life, age measured from the newest candidate.

## Done
- Branch `recency-and-script`.

## Next
- Implement (text.script, segments.script column + backfill, standardize shared with calibrate, recency, damping).
- Measure: SciFact (no regression), LongMemEval-S lead metrics, private sets (outside the repo).
