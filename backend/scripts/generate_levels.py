"""Calibrate, solve, validate, and save puzzles using the active Google vectors."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# These compact vector products are often faster without a large BLAS pool.
# Explicit environment settings always win.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from backend.app.config import Settings
from backend.app.engine.embeddings import EmbeddingStore
from backend.app.engine.level_generator import LevelGenerator
from backend.app.engine.physics import PhysicsConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=settings.model_path)
    parser.add_argument("--vocabulary", type=Path, default=settings.vocabulary_path)
    parser.add_argument("--denylist", type=Path, default=settings.blocked_path)
    parser.add_argument("--output", type=Path, default=settings.levels_path)
    parser.add_argument("--count-per-difficulty", "--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--momentum", type=float, default=settings.momentum)
    parser.add_argument("--impulse", type=float, default=settings.impulse)
    parser.add_argument("--movement-model", choices=["direction", "displacement"], default=settings.movement_model)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=settings.model_binary)
    parser.add_argument("--model-limit", type=int, default=settings.model_limit)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        store = EmbeddingStore(args.model, args.vocabulary, denylist_path=args.denylist, binary=args.binary, limit=args.model_limit).load()
        config = PhysicsConfig(momentum=args.momentum, impulse=args.impulse, movement_model=args.movement_model)
        generator = LevelGenerator(store, config, beam_width=args.beam_width, candidate_count=args.candidate_count)
        print(f"Loaded {len(store.words):,} curated Google News words ({store.fingerprint}).", flush=True)
        print(f"Calibrating {args.count_per_difficulty} puzzles per difficulty with bounded beam search…", flush=True)
        levels = generator.generate(count_per_difficulty=args.count_per_difficulty, seed=args.seed)
        if not all(generator.validate(level) for level in levels):
            raise ValueError("A generated level failed authoritative solution replay.")
        catalog = {
            "schema_version": 1,
            "embedding_fingerprint": store.fingerprint,
            "model": "word2vec-google-news-300",
            "curated_words": len(store.words),
            "physics": asdict(config),
            "generation": {
                "seed": args.seed,
                "beam_width": args.beam_width,
                "candidate_count": args.candidate_count,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.monotonic() - started, 2),
                "human_playtested": False,
            },
            "levels": [level.to_dict() for level in levels],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        for difficulty in ("easy", "medium", "hard"):
            selected = [level for level in levels if level.difficulty == difficulty]
            lengths = Counter(len(level.solution) for level in selected)
            print(f"{difficulty}: {len(selected)} validated levels, threshold {selected[0].threshold:.4f}, "
                  f"solution lengths {dict(sorted(lengths.items()))}", flush=True)
        print(f"Saved {len(levels)} levels to {args.output} in {time.monotonic() - started:.1f}s.", flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Level generation failed: {exc}\n")


if __name__ == "__main__":
    main()
