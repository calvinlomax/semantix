# Verification — October 13, 2026

Verified on macOS Apple Silicon with Python 3.12, Node 22, and installed Google Chrome.

- **30 Python tests passed**, including optional real Google News integration.
- **5 frontend unit tests passed** for completion scoring, daily deduplication, UTC streaks, and spoiler-free sharing.
- **4 Playwright tests passed** across desktop (1440 px) and mobile (390 px) browser configurations.
- **Production frontend build passed**, including strict TypeScript checks.

Browser tests used the actual FastAPI server and prepared Google vectors. They exercised invalid-word rejection without move loss, complete victories, move-limit defeats, deterministic retry, hint consumption, difficulty selection, map controls, help/settings dialogs, reload persistence, daily practice deduplication, reduced-motion preferences, and mobile overflow. Initial desktop/mobile and free-play screenshots were inspected; overlapping map labels and small mobile text were corrected, then browser tests and the build passed again.

The official `word2vec-google-news-300` gzip archive was downloaded from Gensim's release repository. Preparation streamed its 3,000,000 vectors and retained 2,674 of 2,676 curated words, preserving exact original float32 values in all 300 dimensions. The unavailable words were `doughnut` and `grey`. The prepared native vector sidecar is approximately 3.1 MB. The ignored original archive remains available locally for full-model conversion.

All 36 saved level solutions were replayed with the authoritative physics. Calibration details and current thresholds are in `data/levels/README.md`. Difficulty has solver validation but has not undergone broad human playtesting.

Reproduce:

```sh
SEMANTIX_TEST_GOOGLE=1 .venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

The Python test runner emits one dependency deprecation warning from Starlette's test client; no test fails. Browser checks require installed Google Chrome and prepared model/catalog files. Without the model, ordinary Python tests still run using explicit synthetic 300-dimensional fixtures, with the two integration tests skipped.
