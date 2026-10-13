# SEMANTIX

**Navigate the meaning of words.** A playable semantic navigation puzzle powered by Google's original **Google News Word2Vec**, with a React interface and an authoritative Python game server.

Start at one word and steer toward another by entering related words. Each word pulls your position toward its embedding; a little momentum carries into your next move. Reach the target's semantic neighborhood before your moves run out. You cannot enter the target itself. Other words may be repeated. Unknown words never spend a move.

## Run locally

Use **Python 3.11–3.13** (3.12 recommended) and **Node 22.12+**. The following commands run from this repository's root on macOS, including Apple Silicon:

```sh
# If needed, install Python 3.12 and Node with Homebrew.
brew install python@3.12 node

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
npm --prefix frontend install
cp .env.example .env
```

All Python packages stay in the virtual environment. Do not install into a system-managed Python environment. The `.env` copy is optional: defaults match the commands below. Existing prepared weights and catalog can be reused; model preparation is a one-time step.

### Prepare the genuine embeddings

For a fresh checkout, download the official Google News model and retain the curated gameplay vocabulary:

```sh
.venv/bin/python -m backend.scripts.prepare_model \
  --download \
  --vocabulary data/vocabulary/words.txt \
  --output data/models/google-news-300.kv
```

The download is about **1.7 GB**. Curated extraction streams the original binary and retains its exact float32 values. It does not train, approximate, generate, or replace embeddings. The resulting model is only a few MB, while keeping all **300 dimensions**. `data/models/google-news-300.kv.manifest.json` records source, retained vocabulary, missing words, and preparation metadata. The full downloaded archive is cached in ignored `data/downloads/`.

If you already have the original binary or gzip archive:

```sh
.venv/bin/python -m backend.scripts.prepare_model \
  --source data/models/word2vec-google-news-300.gz \
  --vocabulary data/vocabulary/words.txt \
  --output data/models/google-news-300.kv
```

To prepare **all approximately three million original words and phrases**, omit `--vocabulary`. Allow roughly **8–12 GB of free memory** for conversion and several GB of disk space for the resulting native model. A machine with 16 GB RAM or more is recommended for the full workflow. The original float32 vector matrix alone occupies about 3.6 GB; token indexes and temporary allocations add memory. Curated preparation is suitable for modest hardware. In either mode, puzzle generation only uses the curated familiar-word list. Full-model lookup additionally accepts available lowercase alphabetic words that pass validation.

The script uses Gensim's downloader for `word2vec-google-news-300`, or `KeyedVectors.load_word2vec_format(..., binary=True)` for full original-file conversion. It saves the native `.kv` metadata and a separate `.kv.vectors.npy` array. **Keep these together**: only this native format supports the application's `KeyedVectors.load(..., mmap='r')`; the original binary is never passed to an unsupported mmap API. Set `SEMANTIX_MODEL_BINARY=true` to load an original binary directly, with the associated RAM cost. Optional `SEMANTIX_MODEL_LIMIT` restricts original-file loading to its first N entries and may exclude desired words.

### Generate validated levels

```sh
.venv/bin/python -m backend.scripts.generate_levels --count 12 --seed 42
```

This creates `data/levels/catalog.json` with real-model calibration, reproducible solution certificates, and projection landmarks. The catalog is small and can be versioned; downloaded weights cannot. Changing the vocabulary, model, momentum, impulse, or movement model requires regenerating the catalog. The server checks its model fingerprint, physics settings, and solution certificates before accepting it. If no catalog exists, first launch deterministically generates a small nine-level catalog in memory; save an explicit catalog for consistent daily puzzles across installations.

### Start the app

Run in two terminals at the repository root:

```sh
# Terminal 1
make backend
# Equivalent: .venv/bin/python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

```sh
# Terminal 2
make frontend
# Equivalent: npm --prefix frontend run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to the backend. API documentation is at **http://127.0.0.1:8000/docs**. The backend starts without loading the model; the first game initializes it once. Missing weights produce an actionable setup screen and structured 503 response, never synthetic gameplay.

## Playing

- **Daily challenge:** one deterministic medium puzzle per UTC date, six moves. The same catalog produces the same daily puzzle for every player. A completed result is scored once per date in this browser; practice retries do not overwrite it.
- **Free play:** unlimited rounds selected from the validated catalog, with easy/medium/hard difficulty and deterministic retry.
- **Easy:** eight moves, larger target neighborhood, three hints. **Medium:** six moves and one hint. **Hard:** five moves, a smaller neighborhood, and no hints.
- Enter a related word and press Enter or the arrow button. Momentum can make a word help or hurt depending on your route. The move history shows the exact similarity change.
- Drag or zoom the semantic map and use its reset control. The map is an approximation: the numerical cosine similarity is authoritative. It is not a probability or a literal physical distance.
- Help explains the rules. Settings control motion. Statistics and completed results persist in this browser. The share summary contains move outcomes without revealing words.

## Architecture

```text
frontend/                 React + TypeScript + Vite
  src/components/         Stable D3 map, accessible dialogs and game UI
  src/stores/             Zustand game, preferences, local statistics
  src/lib/                API contract, statistics, sharing
  src/styles/             Tailwind + custom dark visual design
backend/
  app/main.py             FastAPI entry point and structured error handlers
  app/models/             Strict Pydantic request validation
  app/services/           Authoritative sessions, daily selection, bounded store
  app/engine/             Embeddings, physics, scoring, PCA, level search
  scripts/                Model conversion and calibrated level generation
  tests/                  Synthetic 300-D unit/API tests and real-model integration
data/vocabulary/          Curated familiar words and an additional denylist
data/levels/              Validated catalog and calibration metadata
```

The frontend uses Tailwind CSS, Motion, D3, Zustand, and Lucide icons. The backend uses FastAPI, Pydantic, NumPy, SciPy, Gensim, and Uvicorn. Only map coordinates, words, scores, and round metadata leave the server. No embedding matrix or solution certificate is sent to the browser.

`API_CONTRACT.md` documents request/response shapes. Main endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Liveness without loading weights |
| `GET /api/model/status` | Loading state and active vocabulary count |
| `POST /api/model/load` | Explicitly initialize weights and levels |
| `GET /api/game/daily` | Start today's UTC challenge |
| `POST /api/game/new` | New free-play round or exact retry |
| `POST /api/game/move` | Validate and commit one authoritative move |
| `GET /api/game/history/{game_id}` | Resume an existing session |
| `POST /api/game/hint` | Spend a hint, without spending a move |

Game identifiers are opaque random tokens. Clients cannot provide positions, velocities, outcomes, or move counts. A per-session lock serializes updates; optional `expected_move` detects stale clients and `request_id` makes request retries idempotent. Invalid words, completed games, missing sessions, and unavailable models return structured errors. Session storage is bounded (default 1,000 games, 24-hour lifetime), isolated behind a store interface, and deliberately single-process.

## Mathematics

All gameplay calculations occur in the original normalized 300-dimensional space. For position `p`, previous velocity `v`, normalized entered-word embedding `w`, and normalized target `g`:

```python
direction = normalize(w - p)
v_next = 0.35 * v + 0.45 * direction
p_next = normalize(p + v_next)
similarity = clip(dot(p_next, g), -1, 1)
distance = 1 - similarity
```

A zero direction contributes no impulse. A perfectly cancelling update keeps the previous valid position; zero or non-finite embeddings are rejected. Victory uses `similarity >= threshold`. Move efficiency is the fraction of the remaining similarity gap closed by that move, clamped to `[-1, 1]`; a regression is negative. All mechanics and retries are deterministic for identical states and inputs.

The configurable `displacement` alternative replaces the direction origin with the previously submitted word embedding: `normalize(w - previous_word)`. Set `SEMANTIX_MOVEMENT_MODEL=displacement` and regenerate levels to compare it. The default movement has the requested 0.35 momentum. Additional gravity, forbidden words/regions, boosted impulses, and tighter precision thresholds have modular engine support; standard generated levels keep advanced modifiers disabled. These require separate calibration/playtesting before being enabled in a game mode.

Each round fits PCA once to a fixed set of start, target, and related landmark embeddings. A deterministic SVD sign convention avoids arbitrary mirrored retries. Positions and submitted words use the same retained transform for the whole round. The 2D projection never influences scoring, physics, solvability, or victory.

## Level calibration

The generator reads a curated vocabulary rather than sampling all news artifacts and phrases. Easy levels draw from actual near neighbors of their target. Medium and hard pools use measured quantiles from a seeded sample of real pair similarities; hard certificates require at least four solver moves. A bounded beam search explores candidate words near the target and its tangent direction while simulating the actual momentum/impulse equations. It derives thresholds from observed best trajectories at increasing search depths, with increasingly tight neighborhoods for harder levels.

Levels must have a replayable solution within the move limit. Already-solved, one-shot, and unreachable candidates are rejected; direct target entry is excluded. Selection considers shortest found route and the number of alternatives retained in the beam. Feedback bands come from observed route similarity quantiles. Every catalog entry records the beam parameters, initial separation band, solution curve, solution length, alternatives found, and model fingerprint. `human_playtested: false` clearly distinguishes computational calibration from human difficulty testing.

Beam search provides a **feasible route**, not a proof of globally shortest play or a guarantee that every route can still win. It is bounded for predictable generation time. Expanding the vocabulary/catalog and human playtesting remain useful ways to improve variety and balance.

## Verification

```sh
# Synthetic fixtures only: no model download needed.
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build

# Optional: complete API rounds using the genuine prepared Google vectors.
SEMANTIX_TEST_GOOGLE=1 .venv/bin/python -m pytest -m integration -q

# End-to-end checks use installed Google Chrome and the prepared real model/catalog.
# Playwright starts both local servers automatically if they are not running.
npm --prefix frontend run test:e2e
```

Tests cover normalization and zero vectors, lookup and unknown words, cosine bounds, momentum and deterministic movement, alternate forces, calibrated search and solution replay, fixed PCA projection, strict API validation, authoritative win/loss transitions, move budgets, stale/concurrent requests, retry identity, UTC daily selection, bounded session expiry, local scoring/streaks, and the rendered gameplay flow. Real-model tests are opt-in so ordinary tests stay fast and offline.

See [VERIFICATION.md](VERIFICATION.md) for the completed local verification: 30 Python tests, five frontend unit tests, four desktop/mobile browser tests, and the production build passed with genuine Google vectors.

## Performance and configuration

`.env.example` lists model paths, original binary limits, physics, allowed local origins, and session bounds. Weights load once per backend process, with native vector arrays memory-mapped read-only. Curated embeddings are normalized once into a compact matrix; other valid lookups use a bounded cache. Gameplay performs 300-dimensional arithmetic and reuses projection anchors. Search scans only the curated matrix, never all three million tokens.

Use **one Uvicorn worker** with the in-memory store. For multiple workers or durable sessions, replace `GameStore` with shared storage and keep moves atomic. Backend restarts expire active sessions; completed browser statistics survive. The dev frontend uses Vite's `/api` proxy; an independently hosted production build needs equivalent same-origin reverse proxy routing and a persistent Python backend. `npm --prefix frontend run build` outputs `frontend/dist/`.

## Known limitations

- The included catalog is finite, so free-play and daily puzzles eventually repeat. Generate a larger catalog for more variety. Installations must share both model/vocabulary and the exact catalog for identical daily selection.
- Difficulty and temperature bands are measured against genuine vectors and solver trajectories, but have not undergone a broad human playtest.
- A reduced vocabulary rejects words absent from that subset. Prepare the full model for broader input coverage. News-trained embeddings can encode historical biases; the curated word list limits puzzle content but cannot remove those biases from the vectors.
- Local statistics are browser metadata, not an account or an anti-cheat service. Clearing browser storage clears them. A daily result is deduplicated locally; there is no global leaderboard.
- Only standard generated levels are enabled by default. Additional mechanics are experimental engine capabilities.
- The 2D map necessarily distorts 300-dimensional relationships; numerical cosine feedback remains authoritative.

Model documentation: [Google News model in Gensim's registry](https://github.com/RaRe-Technologies/gensim-data), [Gensim KeyedVectors and native memory mapping](https://radimrehurek.com/gensim/models/keyedvectors.html). Word vectors are a separately downloaded model, not bundled source assets.
