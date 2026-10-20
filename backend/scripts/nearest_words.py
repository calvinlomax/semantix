"""Print the nearest prepared Google News words for a CLI query.

Run from the repository root with::

    .venv/bin/python -m backend.scripts.nearest_words

The command uses the same reduced, memory-mapped ``.kv`` model as the app. It
does not load the original Google News binary or use the puzzle target list.
"""

from __future__ import annotations

import argparse

import numpy as np

from backend.app.config import Settings
from backend.app.engine.embeddings import EmbeddingStore, ModelUnavailableError, WordValidationError


def nearest(store: EmbeddingStore, word: str, limit: int = 10) -> list[tuple[str, float]]:
    """Return the highest-cosine model entries for *word*, in stable order."""
    resolved = store.resolve(word)
    model = store._model  # The store deliberately keeps the prepared vectors memory-mapped.
    query = resolved.vector
    norms = np.linalg.norm(model.vectors, axis=1)
    scores = (model.vectors @ query) / np.maximum(norms, 1e-12)
    order = np.argsort(-scores, kind="stable")
    results: list[tuple[str, float]] = []
    for index in order:
        candidate = model.index_to_key[int(index)]
        if candidate == resolved.resolved_word:
            continue
        results.append((candidate, float(scores[index])))
        if len(results) == limit:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the nearest prepared Semantix words.")
    parser.add_argument("--top", type=int, default=10, help="number of words to print (default: 10)")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")

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
        word = input("Enter a word: ").strip()
        results = nearest(store, word, args.top)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    except (ModelUnavailableError, WordValidationError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f'Nearest words for "{word}":')
    for rank, (candidate, score) in enumerate(results, 1):
        print(f"{rank:>2}. {candidate:<24} {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
