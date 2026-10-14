# Deploying the reduced model to Render

Prepare the model locally or in a build job with the Google News archive.
Production loads only the reduced native `.kv`; it never loads or converts the
three-million-entry source. The current artifact contains 73,604 embeddings,
using an 88.3 MB vector sidecar and approximately 1.7 MB of metadata.

## Versioned artifacts

Keep the following together in a versioned object-storage location or a release
asset bundle:

```text
google-news-300.kv
google-news-300.kv.vectors.npy
google-news-300.kv.manifest.json
```

The `.kv` references the adjacent sidecar by its filename. Preserve these names.
The manifest records requested/retained/missing words and SHA-256 checksums for
both model files. Publish all three from the same preparation run. The original
archive, model files, and WordNet corpus remain ignored in Git. The generated
text vocabulary files and level catalog are small enough to version normally.
Even when an individual reduced file fits a Git host's limit, artifact storage
avoids adding large binary history to the source repository.

## Native Python service with build-time download

Use the repository root as the Render service root. Configure download URLs as
Render environment variables (secret values for authenticated/signed URLs):

```text
SEMANTIX_MODEL_KV_URL=<URL to the versioned .kv>
SEMANTIX_MODEL_VECTORS_URL=<URL to the matching .kv.vectors.npy>
SEMANTIX_MODEL_MANIFEST_URL=<URL to the matching .kv.manifest.json>
SEMANTIX_MODEL_PATH=data/models/google-news-300.kv
SEMANTIX_VOCABULARY_PATH=data/vocabulary/allowed_guesses.txt
SEMANTIX_TARGET_VOCABULARY_PATH=data/vocabulary/level_words.txt
SEMANTIX_LEVELS_PATH=data/levels/catalog.json
SEMANTIX_NLTK_DATA=data/nltk_data
SEMANTIX_MODEL_BINARY=false
```

Leave `SEMANTIX_MODEL_LIMIT` unset. Install only runtime requirements; wordfreq
is needed to regenerate source lists, not to serve guesses. A build script can
run these commands with shell error handling enabled (`set -e`):

```sh
python -m pip install -r backend/requirements.txt
python -m backend.scripts.prepare_nlp
mkdir -p data/models
curl --fail --location "$SEMANTIX_MODEL_KV_URL" --output data/models/google-news-300.kv
curl --fail --location "$SEMANTIX_MODEL_VECTORS_URL" --output data/models/google-news-300.kv.vectors.npy
curl --fail --location "$SEMANTIX_MODEL_MANIFEST_URL" --output data/models/google-news-300.kv.manifest.json
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('data/models')
manifest = json.loads((root / 'google-news-300.kv.manifest.json').read_text())
for name in ('google-news-300.kv', 'google-news-300.kv.vectors.npy'):
    expected = manifest['artifacts'][name]
    path = root / name
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
        raise SystemExit(f'Artifact checksum mismatch: {name}')
print('Reduced model artifacts verified.')
PY
```

Only download artifacts produced by your trusted preparation pipeline; Gensim
native metadata uses Python serialization. Startup command:

```sh
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
```

Set the health-check path to `/api/health`. After deployment, warm the model with
`POST /api/model/load`, then inspect `GET /api/model/status`: it should report
`reduced: true`, `memory_mapped: true`, a broad `vocabulary_size`, and a much
smaller `target_vocabulary_size`. One worker is required for the in-memory game
store. Restarts expire active rounds; browser statistics survive. Serve the
frontend's `dist/` with `/api` forwarded to this backend; a separate frontend
domain also requires the appropriate CORS origins and API routing.

Render supplies `PORT` and supports secret environment variables; its regular
service filesystem is ephemeral. Build artifacts should therefore be fetched
reproducibly from their versioned storage on each deploy. See [web services](https://render.com/docs/web-services),
[deployment behavior](https://render.com/docs/deploys), and
[environment variables](https://render.com/docs/configure-environment-variables).

## Optional persistent disk

A persistent disk can hold prepared artifacts across deploys. For example,
mount it at `/var/data` and set
`SEMANTIX_MODEL_PATH=/var/data/semantix/google-news-300.kv` and
`SEMANTIX_NLTK_DATA=/var/data/semantix/nltk_data`. Populate the model and WordNet
data at runtime or through an administrative session before warming the model.
**Render disks are unavailable during build and pre-deploy commands**, so the
build-time download recipe must not target that mount. See [persistent disks](https://render.com/docs/disks).

Local cold initialization on the 8 GB development Mac took about 2.2 seconds and
peaked near 343 MiB RSS. These are development measurements, not a production
capacity guarantee; allow memory headroom for Python, WordNet, request traffic,
and the bounded caches. The reduced vector matrix is mapped read-only, and only
the 2,674 retained target vectors are copied into the normalized search matrix.

Set `SEMANTIX_HINT_VOCABULARY_PATH=data/vocabulary/hint_words.txt` alongside the
guess and target paths. Deploy the generated text list with the application.
This update needs no new model extraction when allowed_guesses.txt is unchanged:
use the existing reduced .kv, its vector sidecar, and preparation manifest.
Never use the original full Google News archive at runtime. Model artifacts
remain ignored by Git. Per-target ranking caches are bounded to 32 targets;
budget additional memory for these arrays and the initial validated-entry index.
