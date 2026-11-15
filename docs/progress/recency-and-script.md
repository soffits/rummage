# Recency term and script-weighted word bonus

Goal: newer items first among plausible matches (search and recall), and cross-language queries decided by meaning
(queue: "Cross-language miss", "Recall ignores how recent a lead is", items 9 and 10).
Worktree: `.worktrees/recency-and-script` (another session works in the main checkout).

## Findings
- Cross-language miss is mostly the meaning score, not the word bonus: the embedding scores same-script passages
  higher for the same query. Per-script z centring helped zh->en but hurt en->zh and zh->zh on a private probe
  (as the old decision found), so it is not shipped; z stays per modality.
- Word bonus: scaled by the share of meaning matches in the query's script (the word channel can't see the
  rest); lookups (IDs) keep full weight. Private probe: en->zh better, zh->zh and zh->en unchanged.
- Recency: additive bonus, exponential decay with a half-life, age measured from the newest candidate.
- LoCoMo (no recency questions): recency costs 0.4-0.8 pt of top-10 evidence (weight 0.5-1), more at 2-3.

## Done
- text.script, segments.script (+ backfill on open), standardize shared by search and calibrate, recency term,
  script-weighted words; bench memory.py: session times, ranking variants, newest-evidence metric.
- Private QA baseline (0.16.1): 0.925.

## Next
- LongMemEval-S knowledge-update + temporal numbers (index in main checkout bench/work/lme-ku-tr) -> pick weight.
- SciFact rummage + ablations. Docs, changelog, PR, release, install, private QA after.
