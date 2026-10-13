from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.config import Settings
from backend.app.engine.embeddings import SyntheticEmbeddings
from backend.app.main import create_app
from backend.app.models.requests import MoveRequest
from backend.app.services.errors import GameError
from backend.app.services.game import GameService, daily_index
from backend.app.services.store import GameStore


@pytest.fixture
def service():
    basis = np.eye(300, dtype=np.float32)
    vectors = {
        "forest": basis[0], "ocean": basis[1], "water": basis[1] + .12 * basis[2],
        "river": basis[1] + .2 * basis[3], "tree": basis[0] + .2 * basis[4],
        "cloud": basis[1] + basis[0], "stone": basis[5], "garden": basis[6],
    }
    embeddings = SyntheticEmbeddings(vectors)
    levels = [SimpleNamespace(
        id=f"test-{difficulty}", difficulty=difficulty, start_word="forest", target_word="ocean",
        move_limit=limit, threshold=.8, solution=["water"] * 4, initial_similarity=0.,
        feedback_bands=[.1, .3, .5, .7], landmark_words=["river", "tree", "cloud"],
        model_fingerprint=embeddings.fingerprint, mechanics={},
    ) for difficulty, limit in [("easy", 8), ("medium", 6), ("hard", 5)]]
    return GameService(Settings(), embeddings=embeddings, levels=levels, day_provider=lambda: "2026-10-13")


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as value:
        yield value


def new(client, difficulty="easy"):
    response = client.post("/api/game/new", json={"difficulty": difficulty})
    assert response.status_code == 200, response.text
    return response.json()


def move(client, game, word, **kwargs):
    return client.post("/api/game/move", json={"game_id": game["id"], "word": word, **kwargs})


def test_health_never_loads_missing_model(tmp_path):
    settings = replace(Settings(), model_path=tmp_path / "absent.kv")
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/model/status").json()["status"] == "unloaded"
        response = client.post("/api/game/new", json={})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert client.get("/api/model/status").json()["status"] == "error"


def test_full_win_and_server_authority(client):
    game = new(client)
    assert game["moves_remaining"] == 8
    assert "solution" not in game and "velocity" not in game
    while game["status"] == "playing":
        response = move(client, game, "water", expected_move=game["moves_used"])
        assert response.status_code == 200, response.text
        game = response.json()
    assert game["status"] == "won"
    assert game["similarity"] >= game["threshold"]
    assert game["moves_used"] == len(game["history"])
    assert len(game["map"]["trail"]) == game["moves_used"] + 1
    assert game["distance"] == pytest.approx(1 - game["similarity"])
    assert move(client, game, "river").status_code == 409


def test_invalid_words_and_target_do_not_consume_moves(client):
    game = new(client)
    for word in ["asdfdoesnotexist", "ocean", "two words", "1234"]:
        response = move(client, game, word)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["message"]
    state = client.get(f"/api/game/history/{game['id']}").json()
    assert state["moves_used"] == 0
    assert state["map"] == game["map"]


def test_move_limit_and_deterministic_retry(client):
    game = new(client, "hard")
    original = game
    for _ in range(5):
        game = move(client, game, "forest").json()
    assert game["status"] == "lost"
    assert game["moves_remaining"] == 0
    assert move(client, game, "river").status_code == 409
    replay = client.post("/api/game/new", json={"retry_game_id": game["id"]}).json()
    assert replay["id"] != game["id"]
    assert replay["level_id"] == original["level_id"]
    assert replay["map"] == original["map"]
    assert replay["similarity"] == original["similarity"]


def test_retry_trajectory_is_identical(client):
    game = new(client)
    retry = client.post("/api/game/new", json={"retry_game_id": game["id"]}).json()
    for word in ["river", "tree", "cloud"]:
        game, retry = move(client, game, word).json(), move(client, retry, word).json()
        assert game["history"] == retry["history"]
        assert game["map"] == retry["map"]


def test_api_rejects_client_state_and_bad_inputs(client):
    game = new(client)
    assert move(client, game, "water", position=[1, 2]).status_code == 422
    assert move(client, game, "").status_code == 422
    assert client.post("/api/game/new", json={"difficulty": "impossible"}).status_code == 422
    assert client.post("/api/game/move", json={"word": "water"}).status_code == 422
    assert client.get("/api/game/history/missing").status_code == 404
    assert move(client, game, "water", power_shot=True).status_code == 400
    assert client.get(f"/api/game/history/{game['id']}").json()["moves_used"] == 0


def test_idempotent_requests_and_stale_move_protection(client):
    game = new(client)
    first = move(client, game, "river", request_id="unique", expected_move=0)
    second = move(client, game, "river", request_id="unique", expected_move=0)
    assert first.json() == second.json()
    assert move(client, game, "tree", request_id="unique").status_code == 409
    assert move(client, game, "tree", expected_move=0).status_code == 409
    assert client.get(f"/api/game/history/{game['id']}").json()["moves_used"] == 1


def test_parallel_moves_only_commit_once(service):
    game = service.new_game("easy")
    def submit(word):
        try:
            return service.move(MoveRequest(game_id=game["id"], word=word, expected_move=0))["moves_used"]
        except GameError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, ["river", "cloud"]))
    assert results.count(1) == 1
    assert results.count("stale_game") == 1
    assert service.history(game["id"])["moves_used"] == 1


def test_daily_identity_fixed_and_retry_preserved(client):
    first, second = client.get("/api/game/daily").json(), client.get("/api/game/daily").json()
    assert first["id"] != second["id"]
    assert first["level_id"] == second["level_id"]
    assert first["daily_date"] == "2026-10-13"
    assert first["difficulty"] == "medium" and first["move_limit"] == 6
    retry = client.post("/api/game/new", json={"retry_game_id": first["id"]}).json()
    assert retry["mode"] == "daily" and retry["daily_date"] == first["daily_date"]
    assert daily_index("2026-10-13", 100) == daily_index("2026-10-13", 100)
    assert len({daily_index(f"2026-10-{day:02d}", 100) for day in range(1, 30)}) > 10


def test_projection_stable_for_entire_round(client):
    game = new(client)
    after = move(client, game, "river").json()
    assert after["map"]["start"] == game["map"]["start"]
    assert after["map"]["target"] == game["map"]["target"]
    assert after["map"]["landmarks"] == game["map"]["landmarks"]


def test_hints_budget_and_no_move_consumption(client):
    game = new(client)
    for remaining in [2, 1, 0]:
        response = client.post("/api/game/hint", json={"game_id": game["id"]})
        assert response.status_code == 200, response.text
        assert response.json()["word"] != game["target_word"]
        assert response.json()["remaining"] == remaining
    assert client.post("/api/game/hint", json={"game_id": game["id"]}).status_code == 400
    assert client.get(f"/api/game/history/{game['id']}").json()["moves_used"] == 0


def test_store_ttl_and_capacity():
    now = [0.]
    store = GameStore(capacity=2, ttl=60, clock=lambda: now[0])
    for key in ["a", "b", "c"]:
        store.put(SimpleNamespace(id=key))
    with pytest.raises(GameError):
        store.get("a")
    assert store.get("b").id == "b"
    now[0] = 60
    with pytest.raises(GameError):
        store.get("b")
