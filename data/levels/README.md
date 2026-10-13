Validated level catalogs are generated with the actual prepared Google News
vectors and the exact gameplay physics:

```sh
python -m backend.scripts.generate_levels \
  --model data/models/google-news-300.kv \
  --output data/levels/catalog.json --count-per-difficulty 12
```

The catalog includes its embedding fingerprint and physics configuration, then
each level's measured initial similarity, calibrated success threshold, and
server-only solution trajectory. Never send solution fields to the browser.

Generation is deterministic for a fixed vocabulary, model, physics, seed, and
search configuration. Beam search is bounded, so an unsuccessful search means
“not validated by this search,” not proof that no solution exists. Re-generate
the catalog after changing embeddings, movement mechanics, or vocabulary.

The prepared catalog contains 36 levels using 2,674 authentic Google News words:

| Difficulty | Levels | Cosine threshold | Shortest routes found | Move budget |
| --- | --- | --- | --- | --- |
| Easy | 12 | 0.640269 | 2–4 moves | 8 |
| Medium | 12 | 0.699895 | 3–4 moves | 6 |
| Hard | 12 | 0.717513 | 4 moves | 5 |

Easy pairs are drawn from actual target neighbors, with initial similarities
0.293–0.442. Medium starts span 0.122–0.233; hard starts span −0.047–0.052.
These are measured values for this catalog, not universal tuning constants.
Regeneration recalibrates thresholds. Search certificates establish feasibility;
they do not establish human difficulty or prove global route optimality.
