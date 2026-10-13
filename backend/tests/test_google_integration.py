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
