"""Fast 300-dimensional fixtures; no model download is required."""

from dataclasses import replace

import numpy as np
import pytest

from backend.app.engine.embeddings import (
    EmbeddingStore, ModelUnavailableError, SyntheticEmbeddings, WordValidationError, clean_word,
)
from backend.app.engine.level_generator import Level, LevelGenerator
from backend.app.engine.physics import PhysicsConfig, is_forbidden, step
from backend.app.engine.projection import Projection
from backend.app.engine.scoring import cosine_similarity, efficiency, feedback, normalize, reached_target, semantic_distance


def axis(index):
    value = np.zeros(300, dtype=np.float32)
    value[index] = 1
    return value


@pytest.fixture(scope="module")
def clustered_embeddings():
    rng = np.random.default_rng(4)
    common = rng.normal(size=300) * 0.2
    centers = rng.normal(size=(12, 300))
    vectors = {}
    for index in range(240):
        word = "term" + chr(97 + index // 26) + chr(97 + index % 26)
        vectors[word] = common + centers[index % 12] * 0.4 + rng.normal(size=300) * 0.5
    return SyntheticEmbeddings(vectors)


def test_normalization_cosine_distance_and_zero_handling():
    np.testing.assert_allclose(normalize(5 * axis(0)), axis(0))
    assert cosine_similarity(axis(0), 4 * axis(0)) == 1
    assert cosine_similarity(axis(0), -axis(0)) == -1
    assert cosine_similarity(axis(0), axis(1)) == 0
    assert semantic_distance(axis(0), axis(1)) == 1
    with pytest.raises(ValueError, match="zero"):
        normalize(np.zeros(300))
    np.testing.assert_array_equal(normalize(np.zeros(300), allow_zero=True), np.zeros(300))
    with pytest.raises(ValueError):
        normalize(np.full(300, np.nan))


def test_lookup_normalization_validation_unknown_and_blocked_words():
    embeddings = SyntheticEmbeddings({"apple": 4 * axis(0), "poison": axis(1)}, blocked=("poison",))
    np.testing.assert_array_equal(embeddings.lookup(" APPLE "), axis(0))
    with pytest.raises(WordValidationError) as caught:
        embeddings.lookup("missing")
    assert caught.value.code == "unknown_word"
    with pytest.raises(WordValidationError) as caught:
        embeddings.lookup("poison")
    assert caught.value.code == "blocked_word"
    for invalid in ("two words", "123", "x", "a" * 25, "<script>", "résumé"):
        with pytest.raises(WordValidationError):
            clean_word(invalid)
    assert not embeddings.matrix.flags.writeable


def test_native_loading_mmap_curated_matrix_and_cached_full_vocabulary_lookup(tmp_path):
    from gensim.models import KeyedVectors

    model = KeyedVectors(vector_size=300)
    words = ["token" + chr(97 + index) for index in range(22)]
    model.add_vectors(words, np.stack([axis(index) * 2 for index in range(22)]))
    path = tmp_path / "test.kv"
    model.save(str(path), separately=["vectors"])
    vocabulary = tmp_path / "words.txt"
    vocabulary.write_text("\n".join(words[:20]))
    denylist = tmp_path / "blocked.txt"
    denylist.write_text(words[-1])
    store = EmbeddingStore(path, vocabulary, denylist_path=denylist)
    assert store.status()["state"] == "not_loaded"
    store.load()
    assert store.status()["ready"]
    assert store.matrix.shape == (20, 300)
    np.testing.assert_allclose(np.linalg.norm(store.matrix, axis=1), 1)
    # Legal available words outside the puzzle allowlist remain playable.
    np.testing.assert_array_equal(store.lookup(words[-2]), axis(20))
    assert store.lookup(words[-2]) is store.lookup(words[-2])
    with pytest.raises(WordValidationError):
        store.lookup(words[-1])
    assert store.load() is store
    assert len(store.fingerprint) == 24


def test_missing_model_fails_clearly_without_fallback(tmp_path):
    store = EmbeddingStore(tmp_path / "missing.kv", tmp_path / "words.txt")
    with pytest.raises(ModelUnavailableError, match="prepare_model"):
        store.load()
    assert store.status()["state"] == "error"
    assert store.status()["curated_words"] == 0


def test_step_matches_documented_equations_and_carries_momentum():
    position, word = axis(0), axis(1)
    velocity = axis(2) * 0.3
    expected_velocity = 0.35 * velocity + 0.45 * normalize(word - position)
    expected_position = normalize(position + expected_velocity)
    actual, actual_velocity = step(position, velocity, word)
    np.testing.assert_allclose(actual, expected_position, atol=1e-7)
    np.testing.assert_allclose(actual_velocity, expected_velocity, atol=1e-7)
    np.testing.assert_array_equal(step(position, velocity, word)[0], actual)
    assert np.linalg.norm(actual) == pytest.approx(1)
    assert actual_velocity[2] == pytest.approx(0.105)


def test_zero_direction_cancellation_invalid_dimensions_and_configuration():
    position, velocity = step(axis(0), np.zeros(300), axis(0))
    np.testing.assert_array_equal(position, axis(0))
    np.testing.assert_array_equal(velocity, np.zeros(300))
    position, velocity = step(axis(0), np.zeros(300), -axis(0), PhysicsConfig(momentum=0, impulse=1))
    np.testing.assert_array_equal(position, axis(0))
    assert np.isfinite(velocity).all()
    with pytest.raises(ValueError, match="300"):
        step(np.ones(2), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="zero"):
        step(np.zeros(300), np.zeros(300), axis(1))
    with pytest.raises(ValueError):
        PhysicsConfig(momentum=1.5)


def test_displacement_gravity_power_and_forbidden_regions_are_modular():
    p, v, w = axis(0), np.zeros(300), axis(1)
    standard, _ = step(p, v, w)
    displacement, _ = step(p, v, w, PhysicsConfig(movement_model="displacement"), previous_word=axis(2))
    assert not np.allclose(displacement, standard)
    gravity, _ = step(p, v, w, PhysicsConfig(gravity_strength=0.1), attractor=axis(2))
    assert gravity[2] > standard[2]
    power, _ = step(p, v, w, power=True)
    assert cosine_similarity(power, w) > cosine_similarity(standard, w)
    assert is_forbidden("apple", p, forbidden_words=["apple"])
    assert is_forbidden("apple", p, forbidden_vectors=[p], threshold=0.9)
    assert not is_forbidden("apple", p, forbidden_vectors=[w], threshold=0.9)


def test_victory_feedback_and_efficiency():
    assert reached_target(0.7, 0.7)
    assert not reached_target(0.699, 0.7)
    assert feedback(0.7, 0.7, (0.1, 0.2, 0.4, 0.6)) == "Target reached"
    assert feedback(0.5, 0.7, (0.1, 0.2, 0.4, 0.6)) == "Hot"
    assert feedback(-0.2, 0.7, (0.1, 0.2, 0.4, 0.6)) == "Very cold"
    assert efficiency(0.2, 0.4, 0.6) == pytest.approx(0.5)
    assert efficiency(0.4, 0.2, 0.6) == pytest.approx(-1)


def test_projection_is_deterministic_and_fixed_for_trajectory(clustered_embeddings):
    anchors = clustered_embeddings.matrix[:20]
    first, second = Projection.fit(anchors), Projection.fit(anchors)
    before = first.project(anchors[0])
    position, _ = step(anchors[0], np.zeros(300), anchors[1])
    assert len(first.project(position)) == 2
    assert first.project(anchors[0]) == before
    np.testing.assert_array_equal(first.components, second.components)
    np.testing.assert_array_equal(first.mean, second.mean)
    assert not first.components.flags.writeable
    with pytest.raises(ValueError):
        Projection.fit(np.zeros((2, 2)))


def test_generation_is_calibrated_deterministic_and_solution_replays(clustered_embeddings):
    generator = LevelGenerator(clustered_embeddings)
    levels = generator.generate(count_per_difficulty=2, seed=42)
    again = generator.generate(count_per_difficulty=2, seed=42)
    assert [level.to_dict() for level in levels] == [level.to_dict() for level in again]
    assert len(levels) == 6
    thresholds = {level.difficulty: level.threshold for level in levels}
    assert thresholds["easy"] < thresholds["medium"] < thresholds["hard"]
    for level in levels:
        assert generator.validate(level)
        assert level.target_word not in level.solution
        assert 2 <= len(level.solution) <= level.move_limit
        assert level.initial_similarity < level.threshold
        assert level.calibration["method"] == "observed_bounded_beam_search"
        assert Level.from_dict(level.to_dict()) == level
        assert not generator.validate(replace(level, model_fingerprint="another-model"))
        assert not generator.validate(replace(level, solution=(level.target_word,)))
        assert not generator.validate(replace(level, threshold=1.0))


def test_search_and_hint_use_current_position(clustered_embeddings):
    generator = LevelGenerator(clustered_embeddings)
    start, target = clustered_embeddings.words[:2]
    result = generator.search(start, target, 4)
    assert len(result.best_by_depth) == 4
    for depth, node in enumerate(result.best_by_depth, 1):
        assert len(node.path) == depth
        assert target not in node.path
    level = generator.generate(count_per_difficulty=1)[0]
    position = clustered_embeddings.lookup(level.start_word)
    velocity = np.zeros(300)
    word = generator.hint(level, position, velocity)
    assert word != level.target_word
    assert word in clustered_embeddings.words
