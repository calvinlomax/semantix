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
python -m backend.scripts.prepare_nlp
npm --prefix frontend install
cp .env.example .env
```

All Python packages stay in the virtual environment. Do not install into a system-managed Python environment. The `.env` copy is optional: defaults match the commands below. Existing prepared weights and catalog can be reused; model preparation is a one-time step.

`prepare_nlp` downloads the small NLTK WordNet corpus once into ignored `data/nltk_data/`. Set `SEMANTIX_NLTK_DATA` to use another directory. Word recognition then runs locally, without request-time downloads or repeated NLP initialization. If the corpus is missing, word validation returns an actionable setup error.

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
- Desktop starts in **3D**: drag to rotate, scroll/pinch to zoom, and right-drag to pan. Use the reset button or switch to **2D** at any time. Mobile starts in 2D; unavailable WebGL automatically falls back to 2D. Focus the 3D map to rotate with arrow keys, zoom with +/−, or reset with R/Home.
- Settings offer optional camera following, landmark labels, reduced motion, and separate move-preview preferences. Previews start **on in Free Play** and **off in Daily Challenge**. A ghost marker, dashed trajectory, and predicted similarity change show the backend's proposed move without spending a turn.
- The proximity meter shows current cosine similarity and the winning threshold. After each move, feedback reports improvement or regression and remaining moves. Maps approximate 300-dimensional relationships; numerical similarity is authoritative, not a probability or a literal physical distance.
- The first-visit tutorial explains attraction and momentum. Enter submits, the clear button resets input, and retry/new-puzzle controls remain nearby. Move history collapses; completed rounds replay the trajectory, respecting reduced-motion preferences.
- Statistics, preferences, and completed results persist in this browser. The share summary contains move outcomes without revealing words.

### Word recognition

The server first looks for the exact submitted model entry, then applies case/Unicode presentation normalization, then uses NLTK WordNet lemmatization as a fallback. For example, an unavailable `prepared` can use `prepare`, `studies` can use `study`, and `children` can use `child`. An available exact form always keeps its own original embedding. The move history keeps your submitted spelling and shows a subtle notice such as `"prepared" accepted using "prepare"` when another entry is used.

There are no fuzzy spelling matches or synonym substitutions. When multiple base forms are plausible for an isolated word, the server asks for the intended base form rather than guessing: `better`, for example, can mean `good` or `well`. Exact entries remain usable even when their possible lemmas are ambiguous. Unknown words are rejected without spending a move.

Moves, previews, and hints share the same resolution and rule validation. Entering the target directly, or a fallback that resolves to the target's model entry, is prohibited. An exact inflection with its own different embedding remains legal under the existing target rule. Forbidden-word rules and any level that disables repetition compare grammatical families, including exact inflections, to prevent bypasses. Standard levels still allow repeated words.

## Architecture

```text
frontend/                 React + TypeScript + Vite
  src/components/         Stable 2D/3D maps, accessible dialogs and game UI
  src/stores/             Zustand game, preferences, local statistics
  src/lib/                API contract, statistics, sharing
  src/styles/             Tailwind + custom dark visual design
backend/
  app/main.py             FastAPI entry point and structured error handlers
  app/models/             Strict Pydantic request validation
  app/services/           Authoritative sessions, daily selection, bounded store
  app/engine/             Embeddings, physics, scoring, PCA, level search
  scripts/                Model conversion, WordNet setup, calibrated levels
  tests/                  Synthetic 300-D unit/API tests and real-model integration
data/vocabulary/          Curated familiar words and an additional denylist
data/levels/              Validated catalog and calibration metadata
```

The frontend uses Tailwind CSS, Motion, D3, Three.js with React Three Fiber/Drei, Zustand, and Lucide icons. The backend uses FastAPI, Pydantic, NumPy, SciPy, Gensim, NLTK, and Uvicorn. Only map coordinates, words, scores, and round metadata leave the server. No embedding matrix or solution certificate is sent to the browser.

`API_CONTRACT.md` documents request/response shapes. Main endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Liveness without loading weights |
| `GET /api/model/status` | Loading state and active vocabulary count |
| `POST /api/model/load` | Explicitly initialize weights and levels |
| `GET /api/game/daily` | Start today's UTC challenge |
| `POST /api/game/new` | New free-play round or exact retry |
| `POST /api/game/move` | Validate and commit one authoritative move |
| `POST /api/game/preview` | Validate and predict a move without changing the round |
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

Each round fits three-component PCA once to a fixed set of start, target, and related landmark embeddings. A deterministic SVD sign convention avoids arbitrary mirrored retries. Positions and submitted words use the same retained transform for the whole round; the first two coordinates retain the existing 2D projection. Camera movement and trajectory animations operate only on display coordinates. Neither projection influences scoring, physics, solvability, or victory.

## Level calibration

The generator reads a curated vocabulary rather than sampling all news artifacts and phrases. Easy levels draw from actual near neighbors of their target. Medium and hard pools use measured quantiles from a seeded sample of real pair similarities; hard certificates require at least four solver moves. A bounded beam search explores candidate words near the target and its tangent direction while simulating the actual momentum/impulse equations. It derives thresholds from observed best trajectories at increasing search depths, with increasingly tight neighborhoods for harder levels.

Levels must have a replayable solution within the move limit. Already-solved, one-shot, and unreachable candidates are rejected; direct target entry is excluded. Selection considers shortest found route and the number of alternatives retained in the beam. Feedback bands come from observed route similarity quantiles. Every catalog entry records the beam parameters, initial separation band, solution curve, solution length, alternatives found, and model fingerprint. `human_playtested: false` clearly distinguishes computational calibration from human difficulty testing.

Beam search provides a **feasible route**, not a proof of globally shortest play or a guarantee that every route can still win. It is bounded for predictable generation time. Expanding the vocabulary/catalog and human playtesting remain useful ways to improve variety and balance.

## Verification

```sh
# Synthetic embeddings: no Word2Vec download needed; prepare WordNet once above.
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build

# Optional: complete API rounds using the genuine prepared Google vectors.
SEMANTIX_TEST_GOOGLE=1 .venv/bin/python -m pytest -m integration -q

# End-to-end checks use installed Google Chrome and the prepared real model/catalog.
# Playwright starts both local servers automatically if they are not running.
npm --prefix frontend run test:e2e
```

Tests cover vector normalization, cached exact-first word resolution, regular/irregular inflections, ambiguity and rule protections, original spelling preservation, preview/submission equality and non-mutation, stable 3D/2D projection, and existing physics, calibration, sessions, scoring, and daily behavior. Browser tests cover completed games, preview cancellation, settings persistence, 2D/3D switching, and a WebGL-disabled fallback on desktop/mobile. Real-model tests also replay all 36 saved solution certificates. Ordinary tests run offline once WordNet is prepared.

See [VERIFICATION.md](VERIFICATION.md) for the local verification record and reproducible commands.

## Performance and configuration

`.env.example` lists model/corpus paths, original binary limits, physics, allowed local origins, and session bounds. Weights load once per backend process, with native vector arrays memory-mapped read-only. Curated embeddings are normalized once into a compact matrix; other lookups, lexical analyses, and resolved forms use bounded caches. Gameplay performs 300-dimensional arithmetic and reuses projection anchors. Search scans only the curated matrix, never all three million tokens.

The 3D renderer is loaded on demand and renders only when the scene changes, with capped pixel density and a small landmark set. The 2D fallback avoids downloading the Three.js chunk. Preview requests debounce for 350 ms; replaced requests are aborted and stale responses are discarded. Preview and submission share one backend calculation path, under the session lock.

Use **one Uvicorn worker** with the in-memory store. For multiple workers or durable sessions, replace `GameStore` with shared storage and keep moves atomic. Backend restarts expire active sessions; completed browser statistics survive. The dev frontend uses Vite's `/api` proxy; an independently hosted production build needs equivalent same-origin reverse proxy routing and a persistent Python backend. `npm --prefix frontend run build` outputs `frontend/dist/`.

## Known limitations

- The included catalog is finite, so free-play and daily puzzles eventually repeat. Generate a larger catalog for more variety. Installations must share both model/vocabulary and the exact catalog for identical daily selection.
- Difficulty and temperature bands are measured against genuine vectors and solver trajectories, but have not undergone a broad human playtest.
- A reduced vocabulary accepts additional grammatical forms only when a safe base form exists in that subset. Words without an embedding or an unambiguous available base are rejected. Prepare the full model for broader input coverage. News-trained embeddings can encode historical biases; the curated word list limits puzzle content but cannot remove those biases from the vectors.
- Local statistics are browser metadata, not an account or an anti-cheat service. Clearing browser storage clears them. A daily result is deduplicated locally; there is no global leaderboard.
- Only standard generated levels are enabled by default. Additional mechanics are experimental engine capabilities.
- Both 2D and 3D maps distort 300-dimensional relationships; numerical cosine feedback remains authoritative. Dense labels can overlap as the 3D camera rotates; landmark labels can be hidden in Settings.

Model documentation: [Google News model in Gensim's registry](https://github.com/RaRe-Technologies/gensim-data), [Gensim KeyedVectors and native memory mapping](https://radimrehurek.com/gensim/models/keyedvectors.html). Word vectors are a separately downloaded model, not bundled source assets.
