"""Opt in with SEMANTIX_INTEGRATION_MODEL=/path/to/google-news-300.kv."""

import os
from pathlib import Path

import numpy as np
import pytest

from backend.app.engine.embeddings import EmbeddingStore
from backend.app.engine.level_generator import LevelGenerator
from backend.app.engine.physics import step
from backend.app.engine.scoring import cosine_similarity


@pytest.mark.integration
@pytest.mark.skipif(not (os.getenv("SEMANTIX_INTEGRATION_MODEL") or os.getenv("SEMANTIX_TEST_GOOGLE") == "1"), reason="Google News integration model not requested")
def test_real_google_news_lookup_and_solved_game():
    root = Path(__file__).resolve().parents[2]
    store = EmbeddingStore(os.getenv("SEMANTIX_INTEGRATION_MODEL", str(root / "data/models/google-news-300.kv")), root / "data/vocabulary/words.txt",
                           denylist_path=root / "data/vocabulary/blocked.txt")
    store.load()
    assert store.lookup("king").shape == (300,)
    assert cosine_similarity(store.lookup("king"), store.lookup("queen")) > cosine_similarity(store.lookup("king"), store.lookup("banana"))
    generator = LevelGenerator(store)
    for level in generator.generate(count_per_difficulty=1):
        position = store.lookup(level.start_word)
        velocity = np.zeros(300)
        previous = position
        for word in level.solution:
            vector = store.lookup(word)
            position, velocity = step(position, velocity, vector, previous_word=previous)
            previous = vector
        assert cosine_similarity(position, store.lookup(level.target_word)) >= level.threshold
