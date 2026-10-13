# Verification — October 13, 2026

Verified on macOS Apple Silicon with Python 3.12, Node 22, and installed Google Chrome.

- **84 Python tests passed**, including optional real Google News integration and all 36 catalog solutions.
- **9 frontend unit tests passed** for completion scoring, daily deduplication, UTC streaks, spoiler-free sharing, preview preferences, and preserved submitted words.
- **10 Playwright tests passed** across desktop (1440 px) and mobile (390 px) browser configurations.
- **Production frontend build passed**, including strict TypeScript checks.

Browser tests used the actual FastAPI server and prepared Google vectors. They exercised invalid-word rejection without move loss, complete victories, move-limit defeats, deterministic retry, hint consumption, difficulty selection, map controls, help/settings dialogs, reload persistence, daily practice deduplication, reduced-motion preferences, and mobile overflow. Additional coverage verifies the first-time tutorial, collapsible history and replay, original `Prepared` spelling with the `prepare` fallback, identical preview/submission coordinates, default-off daily previews, cancelled stale previews, 2D/3D switching, and play with WebGL disabled. Initial desktop/mobile and free-play screenshots were inspected.

Backend tests cover regular/irregular verbs and plurals, capitalization/Unicode presentation, exact embedding preference, ambiguous and unknown forms, bounded resolution caches, missing WordNet setup, target/forbidden/repetition protections, shared hint validation, and stable three-component PCA. Repeated previews preserve session position, velocity, budgets, history, and request records. Word-marker tooltips use word-to-target similarity; trajectory markers use the resulting player similarity.

The official `word2vec-google-news-300` gzip archive was downloaded from Gensim's release repository. Preparation streamed its 3,000,000 vectors and retained 2,674 of 2,676 curated words, preserving exact original float32 values in all 300 dimensions. The unavailable words were `doughnut` and `grey`. The prepared native vector sidecar is approximately 3.1 MB. The ignored original archive remains available locally for full-model conversion.

All 36 saved level solutions were replayed through authoritative move validation with normalization enabled. The embedding fingerprint remains `92726a53d85e228a6a5d3bd7`; physics, catalog entries, and thresholds were preserved. Existing exact inflections retain their distinct vectors and prior target-rule behavior, while a fallback to the target entry is rejected. Calibration details and current thresholds are in `data/levels/README.md`. Difficulty has solver validation but has not undergone broad human playtesting.

Reproduce:

```sh
# One-time setup after installing backend dependencies:
.venv/bin/python -m backend.scripts.prepare_nlp

SEMANTIX_TEST_GOOGLE=1 .venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

The Python test runner emits one dependency deprecation warning from Starlette's test client. Vite reports a size warning for the lazy 3D chunk (approximately 974 KB minified / 264 KB gzip); 2D play does not load it. No test or build fails. Browser checks require installed Google Chrome and prepared model/catalog files. Without the model, ordinary Python tests use explicit synthetic 300-dimensional fixtures, with the three integration tests skipped. WordNet must still be prepared once; runtime requests never download corpus data.
