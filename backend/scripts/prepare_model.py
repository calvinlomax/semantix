"""Prepare authentic Google News vectors in Gensim's memory-mappable format.

The curated extraction path streams the original binary, avoiding the several
gigabytes of RAM needed to load every original word. No vectors are synthesized.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "word2vec-google-news-300"
SOURCE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/tag/word2vec-google-news-300"
DEFAULT_VOCABULARY = PROJECT_ROOT / "data/vocabulary/allowed_guesses.txt"
MAX_GUESSES = 150_000


def read_vocabulary(path: Path) -> list[str]:
    """Read our explicitly curated allowlist, ignoring comments and duplicates."""
    words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(word for word in words if word and not word.startswith("#")))


def read_exact(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("The Word2Vec download is incomplete: a vector was truncated.")
    return data


def extract_curated(source: Path, words: list[str]) -> tuple[list[str], np.ndarray, dict]:
    """Stream the canonical Google binary; preserve exact original float32 values."""
    remaining = set(words)
    selected: dict[str, np.ndarray] = {}
    opener = gzip.open if source.suffix == ".gz" else open
    started = time.monotonic()
    with opener(source, "rb") as stream:
        try:
            total, dimension = map(int, stream.readline().split())
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Expected a Word2Vec binary file beginning with '<count> 300'.") from exc
        if dimension != 300:
            raise ValueError(f"Expected 300-dimensional Google News vectors; found {dimension}.")
        if total < 1:
            raise ValueError("The Word2Vec binary contains no vectors.")
        for index in range(total):
            token = bytearray()
            while True:
                character = stream.read(1)
                if not character:
                    raise ValueError("The Word2Vec download is incomplete: a token was truncated.")
                if character == b" ":
                    break
                if character not in (b"\n", b"\r"):
                    token.extend(character)
            raw = read_exact(stream, dimension * 4)
            # The original Google News file stores little-endian IEEE float32.
            word = token.decode("utf-8", errors="replace")
            if word in remaining:
                vector = np.frombuffer(raw, dtype="<f4").copy()
                if np.isfinite(vector).all() and np.linalg.norm(vector) > 0:
                    selected[word] = vector
                remaining.remove(word)
                if not remaining:
                    break
            if (index + 1) % 250_000 == 0:
                print(f"Scanned {index + 1:,}/{total:,} tokens; retained {len(selected):,} words.", flush=True)
    retained = [word for word in words if word in selected]
    if not retained:
        raise ValueError("None of the curated words were found in the source model.")
    matrix = np.stack([selected[word] for word in retained]).astype(np.float32, copy=False)
    return retained, matrix, {
        "original_word_count": total,
        "scanned_word_count": index + 1,
        "requested_word_count": len(words),
        "retained_word_count": len(retained),
        "missing_words": [word for word in words if word not in selected],
        "extraction_seconds": round(time.monotonic() - started, 2),
    }


def download_source(cache_dir: Path) -> Path:
    # Set this before importing gensim.downloader; its default is a home cache.
    os.environ["GENSIM_DATA_DIR"] = str(cache_dir.resolve())
    import gensim.downloader as api

    cache_dir.mkdir(parents=True, exist_ok=True)
    api.BASE_DIR = str(cache_dir.resolve())
    print("Downloading Google's original Google News vectors through Gensim (about 1.7 GB).", flush=True)
    return Path(api.load(MODEL_NAME, return_path=True))


def prepare(source: Path, output: Path, vocabulary: Path | None = None) -> dict:
    from gensim.models import KeyedVectors

    if not source.is_file():
        raise FileNotFoundError(f"Source model not found: {source}. Supply --source or --download.")
    if output.suffix != ".kv":
        raise ValueError("Use a .kv output filename for the reduced native model.")
    vocabulary = vocabulary or DEFAULT_VOCABULARY
    requested = read_vocabulary(vocabulary)
    if not 1 <= len(requested) <= MAX_GUESSES:
        raise ValueError("Request 1–150,000 words; 50,000–100,000 guess candidates is recommended.")
    words, matrix, detail = extract_curated(source, requested)
    model = KeyedVectors(vector_size=300)
    model.add_vectors(words, matrix)
    del matrix
    manifest = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "source_url": SOURCE_URL,
        "source_file": source.name,
        "source_bytes": source.stat().st_size,
        "dimensions": model.vector_size,
        "vector_dtype": str(model.vectors.dtype),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "reduced_vocabulary": True,
        "vocabulary_file": vocabulary.name,
        "vocabulary_sha256": hashlib.sha256(vocabulary.read_bytes()).hexdigest(),
        "missing_word_count": len(detail["missing_words"]),
        "format": "gensim-keyedvectors-with-separate-vectors",
        **detail,
    }
    model.semantix_provenance = manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    # Only native Gensim sidecar .npy vectors support mmap='r'. The original
    # Word2Vec binary cannot be memory-mapped by load_word2vec_format.
    # Build and check the entire artifact before replacing the active files.
    # Stop/restart the single backend process when publishing a new model.
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        staged = Path(staging) / output.name
        model.save(str(staged), separately=["vectors"])
        check = KeyedVectors.load(str(staged), mmap="r")
        if not isinstance(check.vectors, np.memmap):
            raise RuntimeError("Prepared vectors were not saved as a memory-mappable sidecar.")
        manifest["artifacts"] = {}
        for path in (staged, Path(str(staged) + ".vectors.npy")):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        del check
        staged_manifest = staged.with_suffix(".kv.manifest.json")
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for path in (Path(str(staged) + ".vectors.npy"), staged, staged_manifest):
            path.replace(output.parent / path.name)
    print(f"Prepared {len(model):,}/{len(requested):,} requested authentic 300-D vectors: {output}", flush=True)
    if detail.get("missing_words"):
        print(f"Skipped {len(detail['missing_words'])} unavailable words (listed in the manifest).", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path, help="Original Google News .bin or .gz binary")
    source_group.add_argument("--download", action="store_true", help="Download the genuine source using gensim.downloader")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/models/google-news-300.kv")
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY, help="Guess candidates to stream from the source; defaults to allowed_guesses.txt. Never loads the full model.")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/downloads")
    args = parser.parse_args()
    try:
        source = download_source(args.cache_dir) if args.download else args.source
        prepare(source, args.output, args.vocabulary)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Model preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
