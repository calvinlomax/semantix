"""Opt in: SEMANTIX_TEST_GOOGLE=1 .venv/bin/python -m pytest -m integration -q."""
import json
import os

from fastapi.testclient import TestClient
import pytest

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.game import GameService

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.getenv("SEMANTIX_TEST_GOOGLE") != "1", reason="Real Google model integration is opt-in")]


def test_real_google_lookup_catalog_and_full_api_games():
    settings = Settings()
    service = GameService(settings)
    service.ensure_ready()
    assert service.embeddings.lookup("king").shape == (300,)
    assert len(service.embeddings.words) >= 100
    with TestClient(create_app(service)) as client:
        assert client.get("/api/model/status").json()["status"] == "ready"
        for difficulty in ("easy", "medium", "hard"):
            game = client.post("/api/game/new", json={"difficulty": difficulty}).json()
            level = next(item for item in service.levels if item.id == game["level_id"])
            for word in level.solution:
                response = client.post("/api/game/move", json={"game_id": game["id"], "word": word, "expected_move": game["moves_used"]})
                assert response.status_code == 200, response.text
                game = response.json()
                if game["status"] == "won":
                    break
            assert game["status"] == "won", json.dumps(game)
            assert game["similarity"] >= game["threshold"]


def test_every_existing_catalog_certificate_remains_playable_with_word_resolution():
    """The normalization update must not invalidate previously calibrated puzzles."""
    from backend.app.models.requests import MoveRequest
    service = GameService(Settings())
    service.ensure_ready()
    levels = service.levels
    for level in levels:
        service.levels = [level]
        game = service.new_game(level.difficulty)
        for word in level.solution:
            game = service.move(MoveRequest(game_id=game["id"], word=word))
            if game["status"] == "won":
                break
        assert game["status"] == "won", level.id


def test_large_guess_model_and_common_words_preserve_curated_answers():
    import numpy as np
    from backend.scripts.build_large_vocab import read_words

    settings = Settings()
    service = GameService(settings)
    service.ensure_ready()
    store = service.embeddings
    status = service.model_status()
    assert 50_000 <= status["vocabulary_size"] <= 150_000
    assert status["requested_guess_count"] > status["vocabulary_size"]
    assert status["missing_guess_count"] == status["requested_guess_count"] - status["vocabulary_size"]
    assert status["target_vocabulary_size"] == len(store.words) < status["vocabulary_size"] / 20
    assert status["reduced"] and status["memory_mapped"]
    targets = set(read_words(settings.target_vocabulary_path))
    assert set(store.words) <= targets <= set(read_words(settings.vocabulary_path))
    assert all(level.start_word in targets and level.target_word in targets for level in service.levels)
    for word in ("computer", "beautiful", "running", "river", "school", "music", "science", "garden", "painting", "memory", "photosynthesis", "thermodynamics", "astronomy", "prepared"):
        resolved = store.resolve(word)
        assert resolved.resolution == "exact", word
        assert resolved.vector.shape == (300,) and np.isfinite(resolved.vector).all()
    # An expanded model must keep original calibrated vectors, not normalized replacements.
    assert store.fingerprint == json.loads(settings.levels_path.read_text())["embedding_fingerprint"]
