# Verification — October 13, 2026

Verified on macOS Apple Silicon with Python 3.12, Node 22, and installed Google Chrome.

- **94 Python tests passed**, including optional real Google News integration and all 36 catalog solutions.
- **9 frontend unit tests passed** for completion scoring, daily deduplication, UTC streaks, spoiler-free sharing, preview preferences, and preserved submitted words.
- **12 Playwright tests passed** across desktop (1440 px) and mobile (390 px) browser configurations.
- **Production frontend build passed**, including strict TypeScript checks.

Browser tests used the actual FastAPI server and prepared Google vectors. They exercised invalid-word rejection without move loss, complete victories, move-limit defeats, deterministic retry, hint consumption, difficulty selection, map controls, help/settings dialogs, reload persistence, daily practice deduplication, reduced-motion preferences, and mobile overflow. Additional coverage verifies the first-time tutorial, collapsible history and replay, original `Prepared` spelling with case normalization to its now-retained exact `prepared` vector, identical preview/submission coordinates, default-off daily previews, cancelled stale previews, 2D/3D switching, and play with WebGL disabled. The vocabulary expansion adds desktop/mobile checks that `photosynthesis` works, puzzle targets remain curated, status reports separate counts, and blocked words still spend no move. Initial desktop/mobile and free-play screenshots were inspected.

Backend tests cover regular/irregular verbs and plurals, capitalization/Unicode presentation, exact embedding preference, ambiguous and unknown forms, bounded resolution caches, missing WordNet setup, target/forbidden/repetition protections, shared hint validation, and stable three-component PCA. Repeated previews preserve session position, velocity, budgets, history, and request records. Word-marker tooltips use word-to-target similarity; trajectory markers use the resulting player similarity.

The vocabulary builder combines wordfreq 3.1.1, WordNet 3.0, and the existing curated seed. It produces 94,037 guess candidates and preserves the ordered 2,676-word target list. An independent rebuild reproduced both vocabulary files and the build manifest byte-for-byte. Preparation streamed the official Google News archive's 3,000,000 vectors and retained **73,604 exact embeddings**, preserving original float32 values in all 300 dimensions. The manifest lists 20,433 missing guess candidates. Among curated targets, 2,674 remain available (`doughnut` and `grey` are absent).

The `.kv` metadata is 1,700,369 bytes, its memory-mapped vector sidecar is 88,324,928 bytes, and the model manifest is approximately 356 KB. The original archive remains ignored and is used only for offline extraction. Runtime rejects original binaries, missing/full-model manifests, and non-memory-mappable production artifacts. The source lists are preparation inputs; tests confirm an absent retained embedding cannot be enabled by editing a text list and that a retained model entry remains usable even if omitted from that list.

Local cold initialization of the expanded reduced model took approximately **2.2 seconds** and peaked near **343 MiB RSS**, measured in a fresh Python process with `resource.getrusage`. The normalized target search matrix remains `(2674, 300)`; the server does not normalize/copy the entire guess model or load the full source. These local measurements are not production load-test results.

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

The Python test runner emits one dependency deprecation warning from Starlette's test client. Vite reports a size warning for the lazy 3D chunk (approximately 974 KB minified / 264 KB gzip); 2D play does not load it. No test or build fails. Browser checks require installed Google Chrome and the expanded prepared model/catalog files. Without the model, ordinary Python tests use explicit synthetic 300-dimensional fixtures, with the four integration tests skipped. WordNet must still be prepared once; runtime requests never download corpus data.
