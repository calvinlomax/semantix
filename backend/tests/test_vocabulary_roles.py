"""Prepared model membership and curated puzzle membership are independent."""
from dataclasses import replace
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from gensim.models import KeyedVectors

from backend.app.config import Settings
from backend.app.engine.embeddings import EmbeddingStore, ModelUnavailableError, WordValidationError
from backend.app.engine.level_generator import Level, LevelGenerator
from backend.app.main import create_app
from backend.app.services.game import GameService
from backend.scripts.prepare_model import prepare


@pytest.fixture
def model_files(tmp_path):
    targets = ["token" + chr(97 + index) for index in range(20)]
    extras = ["computer", "prepared", "prepare", "car", "good", "poison", "poisoned", "albatross"]
    source = tmp_path / "source.bin"
    words = targets + extras
    basis = np.eye(300, dtype=np.float32)
    with source.open("wb") as handle:
        handle.write(f"{len(words)} 300\n".encode())
        for index, word in enumerate(words):
            handle.write(word.encode() + b" " + basis[index].astype("<f4").tobytes() + b"\n")
    prep_words = tmp_path / "prep.txt"
    prep_words.write_text("\n".join(words + ["zzmissingword"]))
    path = tmp_path / "model.kv"
    prepare(source, path, prep_words)
    # Source-list edits cannot grant/revoke runtime model membership.
    guesses = tmp_path / "allowed_guesses.txt"
    guesses.write_text("computer\nzzmissingword\n")
    target_path = tmp_path / "level_words.txt"
    target_path.write_text("\n".join(targets + ["zzmissingword", "poisoned"]))
    blocked = tmp_path / "blocked.txt"
    blocked.write_text("poison\n")
    return path, guesses, target_path, blocked, targets


def store_for(files):
    path, guesses, targets, blocked, _ = files
    return EmbeddingStore(path, guesses, target_vocabulary_path=targets, denylist_path=blocked, require_prepared=True).load()


def test_prepared_model_is_runtime_authority_and_target_matrix_stays_small(model_files):
    store = store_for(model_files)
    assert store.words == tuple(model_files[-1])
    assert store.matrix.shape == (20, 300)
    assert isinstance(store._model.vectors, np.memmap)
    assert store.resolve("albatross").resolution == "exact"  # Missing from source text; present in .kv.
    assert store.resolve("COMPUTER").resolution == "normalized"
    assert store.resolve("cars").resolved_word == "car"  # Missing from both source and .kv.
    assert store.resolve("prepared").resolved_word == "prepared"
    np.testing.assert_array_equal(store.resolve("prepared").vector, store.lookup("prepared"))
    with pytest.raises(WordValidationError) as caught:
        store.resolve("zzmissingword")  # Listed in prep source but absent in Google fixture.
    assert caught.value.code == "unknown_word"
    with pytest.raises(WordValidationError) as caught:
        store.resolve("better")
    assert caught.value.code == "ambiguous_word"
    for word in ("poison", "Poisoned", "poisons"):
        with pytest.raises(WordValidationError) as caught:
            store.resolve(word)
        assert caught.value.code == "blocked_word"


def test_broad_guess_expansion_preserves_target_fingerprint_and_solver_pool(model_files, tmp_path):
    broad = store_for(model_files)
    target_words = model_files[-1]
    small_model = KeyedVectors(vector_size=300)
    small_model.add_vectors(target_words, np.stack([broad._model[word] for word in target_words]))
    small_path = tmp_path / "small.kv"
    small_model.save(str(small_path), separately=["vectors"])
    old = EmbeddingStore(small_path, model_files[2]).load()
    assert old.fingerprint == broad.fingerprint
    np.testing.assert_array_equal(old.matrix, broad.matrix)
    generator = LevelGenerator(broad)
    result = generator.search(target_words[0], target_words[1], 3)
    assert result.best_by_depth
    assert all(set(node.path) <= set(target_words) for node in result.best_by_depth)
    assert set(generator._landmarks(target_words[0], target_words[1])) <= set(target_words)
    invalid = Level("invalid", "easy", target_words[0], "albatross", 8, .7, ("computer",), 0,
                    (.1, .2, .3, .4), (), broad.fingerprint)
    assert not generator.validate(invalid)  # A playable guess cannot become an answer.


def test_status_reports_guesses_separately_from_targets(model_files):
    store = store_for(model_files)
    service = GameService(Settings(), embeddings=store, levels=[])
    with TestClient(create_app(service)) as client:
        status = client.get("/api/model/status").json()
    assert status["vocabulary_size"] == 28
    assert status["target_vocabulary_size"] == 20
    assert status["requested_target_count"] == 22
    assert status["requested_guess_count"] == 29
    assert status["missing_guess_count"] == 1
    assert status["reduced"] and status["memory_mapped"]


def test_production_rejects_full_model_manifest_before_loading(model_files, monkeypatch):
    path = model_files[0]
    manifest_path = path.with_suffix(".kv.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(reduced_vocabulary=False, retained_word_count=3_000_000)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(KeyedVectors, "load", lambda *_args, **_kwargs: pytest.fail("Must not load a full model"))
    with pytest.raises(ModelUnavailableError, match="reduced model preparation manifest"):
        store_for(model_files)


def test_app_rejects_original_binary_and_missing_native_manifest(model_files):
    path, guesses, targets, blocked, _ = model_files
    settings = replace(Settings(), model_path=path.with_name("source.bin"), vocabulary_path=guesses,
                       target_vocabulary_path=targets, blocked_path=blocked)
    with TestClient(create_app(settings=settings)) as client:
        response = client.post("/api/model/load")
        assert response.status_code == 503 and "reduced native .kv" in response.json()["error"]["message"]
    path.with_suffix(".kv.manifest.json").unlink()
    with pytest.raises(ModelUnavailableError, match="manifest"):
        store_for(model_files)


def test_separate_hint_file_intersects_model_and_rejects_blocked_families(model_files, tmp_path):
    path, guesses, targets, blocked, _ = model_files
    hints = tmp_path / 'hints.txt'
    hints.write_text('computer\npoisoned\nzzmissingword\n')
    store = EmbeddingStore(path, guesses, target_vocabulary_path=targets,
                           hint_vocabulary_path=hints, denylist_path=blocked).load()
    assert store.hint_words == ('computer',)
    assert 'computer' not in store.words
    assert store.status()['hint_vocabulary_size'] == 1
    from backend.app.engine.semantic import SemanticIndex
    neighbors = SemanticIndex(store).neighbors(store.words[0])
    assert [word for word, _ in neighbors] == ['computer']
