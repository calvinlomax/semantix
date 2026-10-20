"""Find a high-scoring Semantix route between two CLI words.

This uses the same bounded beam search and 300-dimensional movement physics as
level generation. Candidate moves come from the curated puzzle vocabulary, so
the result is a playable route rather than an arbitrary chain of model tokens.

Run from the repository root::

    .venv/bin/python -m backend.scripts.find_route
"""

from __future__ import annotations

import argparse

from backend.app.config import Settings
from backend.app.engine.embeddings import EmbeddingStore, ModelUnavailableError, WordValidationError
from backend.app.engine.level_generator import LevelGenerator
from backend.app.engine.physics import PhysicsConfig
from backend.app.engine.scoring import cosine_similarity


def main() -> int:
    parser = argparse.ArgumentParser(description="Find a Semantix route between two words.")
    parser.add_argument("--max-moves", type=int, default=8, help="maximum route length (default: 8)")
    parser.add_argument("--beam-width", type=int, default=12, help="candidate paths retained per step (default: 12)")
    parser.add_argument("--candidate-count", type=int, default=64, help="candidate words considered per step (default: 64)")
    args = parser.parse_args()
    if args.max_moves < 1 or args.beam_width < 1 or args.candidate_count < 2:
        parser.error("max-moves, beam-width, and candidate-count must be positive")

    settings = Settings()
    store = EmbeddingStore(
        settings.model_path,
        settings.vocabulary_path,
        denylist_path=settings.blocked_path,
        target_vocabulary_path=settings.target_vocabulary_path,
        hint_vocabulary_path=settings.hint_vocabulary_path,
        require_prepared=True,
        binary=settings.model_binary,
        limit=settings.model_limit,
    )
    try:
        store.load()
        start = input("Start word: ").strip()
        destination = input("Destination word: ").strip()
        start_resolution = store.resolve(start)
        destination_resolution = store.resolve(destination)
        if start_resolution.resolved_word == destination_resolution.resolved_word:
            print("Start and destination resolve to the same word.")
            return 1
        generator = LevelGenerator(
            store,
            PhysicsConfig(momentum=settings.momentum, impulse=settings.impulse,
                          movement_model=settings.movement_model),
            beam_width=args.beam_width,
            candidate_count=args.candidate_count,
        )
        result = generator.search(
            start_resolution.resolved_word,
            destination_resolution.resolved_word,
            args.max_moves,
        )
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    except (ModelUnavailableError, WordValidationError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if not result.best_by_depth:
        print("No route could be found with the current search settings.")
        return 1

    # Every beam layer is sorted by final destination similarity. Select the
    # strongest reachable state, preferring fewer moves when scores tie.
    node = max(enumerate(result.best_by_depth, 1), key=lambda item: (item[1].score, -item[0]))[1]
    target = destination_resolution.vector
    initial_score = cosine_similarity(start_resolution.vector, target)
    print(f'Best route from "{start}" to "{destination}":')
    print(f"Start similarity: {initial_score:.4f}")
    for number, (word, score) in enumerate(zip(node.path, node.similarities), 1):
        print(f"{number:>2}. {word:<24} {score:.4f}")
    print(f"Final similarity: {node.score:.4f}")
    print("Destination is the goal vector and is not used as an intermediate move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
