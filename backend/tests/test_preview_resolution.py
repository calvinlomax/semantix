"""Word resolution and preview use the exact same authoritative validation."""
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.engine.embeddings import SyntheticEmbeddings
from backend.app.main import create_app
from backend.app.services.game import GameService


@pytest.fixture
def setup():
    basis = np.eye(300, dtype=np.float32)
    words = {
        "forest": basis[0], "ocean": basis[1], "prepare": basis[1] + .2 * basis[2],
        "car": basis[1] + .3 * basis[3], "walk": basis[1] + .4 * basis[4],
        "run": basis[1] + .3 * basis[5], "study": basis[1] + .1 * basis[6],
        "child": basis[1] + .3 * basis[7], "go": basis[1] + .4 * basis[8],
        # An exact inflection must keep its different embedding, but share rule keys.
        "walked": basis[9], "tree": basis[10], "river": basis[11],
    }
    embeddings = SyntheticEmbeddings(words)
    level = SimpleNamespace(
        id="morphology", difficulty="easy", start_word="forest", target_word="ocean",
        move_limit=8, threshold=.99, solution=["prepare"] * 6, initial_similarity=0.,
        feedback_bands=[.1, .3, .5, .7], landmark_words=["river", "tree", "car"],
        model_fingerprint=embeddings.fingerprint, mechanics={},
    )
    service = GameService(Settings(), embeddings=embeddings, levels=[level])
    with TestClient(create_app(service)) as client:
        game = client.post("/api/game/new", json={"difficulty": "easy"}).json()
        yield client, service, game


def request(client, endpoint, game, word, **extra):
    return client.post(f"/api/game/{endpoint}", json={"game_id": game["id"], "word": word, **extra})


@pytest.mark.parametrize(("word", "resolved"), [
    ("Prepared", "prepare"), ("running", "run"), ("studies", "study"),
    ("cars", "car"), ("children", "child"), ("went", "go"),
])
def test_normalized_preview_and_submission_match_and_preserve_input(setup, word, resolved):
    client, service, game = setup
    original = service.history(game["id"])
    state = service.store.get(game["id"])
    velocity = state.velocity.copy()
    for _ in range(2):
        response = request(client, "preview", game, word, expected_move=0)
        assert response.status_code == 200, response.text
        preview = response.json()
        assert preview["word"] == word and preview["resolved_word"] == resolved
        assert preview["resolution"] == "lemma"
        assert preview["normalization_message"] == f'"{word}" accepted using "{resolved}"'
    assert service.history(game["id"]) == original
    np.testing.assert_array_equal(state.velocity, velocity)
    assert not state.used_word_keys and not state.requests
    response = request(client, "move", game, word, expected_move=0, request_id="one")
    assert response.status_code == 200, response.text
    after = response.json()
    move = after["history"][0]
    for key in ("word", "resolved_word", "resolution", "normalization_message", "position", "position3d",
                "word_position", "word_position3d", "word_similarity", "similarity_before", "similarity_after", "delta", "efficiency"):
        assert move[key] == preview[key]
    assert after["map"]["player"] == preview["position"]
    assert after["map3d"]["player"] == preview["position3d"]
    assert after["history"][0]["word"] == word
    target = service.embeddings.lookup(state.level.target_word)
    assert move["word_similarity"] == pytest.approx(float(service.embeddings.lookup(resolved) @ target))
    assert move["word_similarity"] != move["similarity_after"]


def test_exact_form_keeps_own_vector_and_normalization_notice_is_absent(setup):
    client, service, game = setup
    exact = request(client, "preview", game, "walked").json()
    base = request(client, "preview", game, "walk").json()
    assert exact["resolved_word"] == "walked" and exact["resolution"] == "exact"
    assert exact["normalization_message"] is None
    assert exact["similarity_after"] != base["similarity_after"]
    capitalized = request(client, "move", game, "WALKED").json()["history"][0]
    assert capitalized["word"] == "WALKED" and capitalized["resolved_word"] == "walked"
    assert capitalized["similarity_after"] == exact["similarity_after"]


@pytest.mark.parametrize("word", ["oceans", "OCEAN", "ｏｃｅａｎ"])
def test_target_variants_rejected_equally_in_preview_and_move(setup, word):
    client, service, game = setup
    for endpoint in ("preview", "move"):
        response = request(client, endpoint, game, word)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "target_word"
    assert service.history(game["id"])["moves_used"] == 0


def test_exact_inflections_cannot_bypass_forbidden_words(setup):
    client, service, game = setup
    level = service.store.get(game["id"]).level
    level.mechanics = {"forbidden_words": ["walk"]}
    for word in ("walked", "walking", "WALKS"):
        for endpoint in ("preview", "move"):
            response = request(client, endpoint, game, word)
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "forbidden_word"


def test_target_restrictions_preserve_distinct_exact_embeddings(setup):
    client, service, game = setup
    service.store.get(game["id"]).level.target_word = "walk"
    # Existing puzzles may use an exact related form; its vector is not walk's.
    assert request(client, "preview", game, "walked").status_code == 200
    assert request(client, "move", game, "walked").status_code == 200
    # An unavailable inflection would resolve to the actual forbidden vector.
    assert request(client, "move", game, "walking").json()["error"]["code"] == "target_word"


def test_repetition_aliases_respected_only_when_repetitions_are_disabled(setup):
    client, service, game = setup
    service.store.get(game["id"]).level.mechanics = {"allow_repeats": False}
    assert request(client, "move", game, "walked").status_code == 200
    for word in ("walk", "walking", "WALKED"):
        for endpoint in ("preview", "move"):
            response = request(client, endpoint, game, word)
            assert response.json()["error"]["code"] == "repeated_word"
    service.store.get(game["id"]).level.mechanics = {}
    assert request(client, "move", game, "walk").status_code == 200


def test_unknown_invalid_and_stale_previews_are_nonmutating(setup):
    client, service, game = setup
    original = service.history(game["id"])
    for word in ("zzzzinvalidwords", "two words", "abc123"):
        preview = request(client, "preview", game, word)
        move = request(client, "move", game, word)
        assert preview.status_code == move.status_code == 400
        assert preview.json() == move.json()
    assert request(client, "preview", game, "car", expected_move=1).status_code == 409
    assert request(client, "preview", game, "car", position=[0, 1]).status_code == 422
    assert service.history(game["id"]) == original


def test_hint_is_subject_to_same_morphological_rules(setup):
    client, service, game = setup
    # Simulate a solver suggesting an exact forbidden inflection.
    service.store.get(game["id"]).level.mechanics = {"forbidden_words": ["walk"]}
    service.generator.hint = lambda *args, **kwargs: "walked"
    hint = client.post("/api/game/hint", json={"game_id": game["id"]})
    assert hint.status_code == 200, hint.text
    word = hint.json()["word"]
    assert not service.embeddings.rule_keys(word) & service.embeddings.rule_keys("walk")
    assert request(client, "preview", game, word).status_code == 200
    assert request(client, "move", game, word).status_code == 200


def test_3d_projection_remains_fixed_and_2d_is_same_basis(setup):
    client, service, game = setup
    initial = game["map3d"]
    for word in ("cars", "prepare", "tree"):
        game = request(client, "move", game, word).json()
        for key in ("start", "target", "landmarks"):
            assert game["map3d"][key] == initial[key]
        assert game["map"]["player"] == game["map3d"]["player"][:2]
        assert len(game["map3d"]["trail"]) == len(game["history"]) + 1
