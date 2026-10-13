"""WordNet-backed lexical fallback; embeddings and physical meaning stay intact."""

import numpy as np
import pytest

from backend.app.engine import embeddings as embedding_module
from backend.app.engine.embeddings import (
    EmbeddingStore, SyntheticEmbeddings, WordValidationError,
)
from backend.app.engine.morphology import MorphologyUnavailableError


def axis(index):
    vector = np.zeros(300, dtype=np.float32)
    vector[index] = 1
    return vector


@pytest.mark.parametrize("submitted,base", [
    ("prepared", "prepare"), ("running", "run"), ("walked", "walk"),
    ("studies", "study"), ("cars", "car"),
    ("ran", "run"), ("went", "go"), ("children", "child"),
    ("mice", "mouse"), ("geese", "goose"), ("teeth", "tooth"),
])
def test_regular_and_irregular_inflections_use_base_embedding(submitted, base):
    store = SyntheticEmbeddings({base: axis(0)})
    result = store.resolve(submitted)
    assert result.original_word == submitted
    assert result.resolved_word == base
    assert result.resolution == "lemma"
    assert base in result.rule_keys
    np.testing.assert_array_equal(result.vector, store.lookup(base))
    assert not result.vector.flags.writeable


def test_exact_inflected_embedding_wins_over_lemma():
    store = SyntheticEmbeddings({"prepared": axis(0), "prepare": axis(1)})
    result = store.resolve("prepared")
    assert result.resolved_word == "prepared"
    assert result.resolution == "exact"
    np.testing.assert_array_equal(result.vector, axis(0))
    assert "prepare" in result.rule_keys


def test_exact_original_capitalization_wins_then_case_normalization():
    store = SyntheticEmbeddings({"Apple": axis(0), "apple": axis(1)})
    exact = store.resolve(" Apple ")
    assert exact.original_word == "Apple"
    assert exact.resolved_word == "Apple"
    assert exact.resolution == "exact"
    np.testing.assert_array_equal(exact.vector, axis(0))
    lowered = store.resolve(" APPLE ")
    assert lowered.original_word == "APPLE"
    assert lowered.resolved_word == "apple"
    assert lowered.resolution == "normalized"
    np.testing.assert_array_equal(lowered.vector, axis(1))
    assert exact.rule_keys == lowered.rule_keys


def test_capitalized_fallback_preserves_original_and_unicode_presentation():
    store = SyntheticEmbeddings({"prepare": axis(0)})
    result = store.resolve("  Prepared  ")
    assert result.original_word == "Prepared"
    assert result.resolved_word == "prepare"
    assert result.resolution == "lemma"
    assert store.resolve("ＰＲＥＰＡＲＥ").resolved_word == "prepare"


@pytest.mark.parametrize("word", ["axes", "leaves", "better", "best"])
def test_ambiguous_forms_require_explicit_base_even_if_only_one_candidate_available(word):
    # Model availability cannot provide grammatical context or choose a sense.
    store = SyntheticEmbeddings({"axe": axis(0), "leaf": axis(1), "good": axis(2)})
    with pytest.raises(WordValidationError, match="possible base form") as caught:
        store.resolve(word)
    assert caught.value.code == "ambiguous_word"


def test_ambiguous_surface_with_its_own_embedding_is_accepted_exactly():
    store = SyntheticEmbeddings({"better": axis(0), "good": axis(1), "well": axis(2)})
    result = store.resolve("better")
    assert result.resolution == "exact"
    assert result.resolved_word == "better"
    np.testing.assert_array_equal(result.vector, axis(0))


@pytest.mark.parametrize("word", ["qzxvunknown", "prepaerd", "carrs"])
def test_unknown_words_do_not_get_fuzzy_spelling_matches(word):
    store = SyntheticEmbeddings({"prepare": axis(0), "car": axis(1)})
    with pytest.raises(WordValidationError, match="recognized base form") as caught:
        store.resolve(word)
    assert caught.value.code == "unknown_word"


@pytest.mark.parametrize("word", ["two words", "car's", "run!", "12", "a", "résumé"])
def test_token_normalization_never_discards_meaningful_punctuation(word):
    store = SyntheticEmbeddings({"run": axis(0), "car": axis(1)})
    with pytest.raises(WordValidationError) as caught:
        store.resolve(word)
    assert caught.value.code == "invalid_word"


def test_rule_keys_match_inflections_even_when_both_have_distinct_exact_vectors():
    store = SyntheticEmbeddings({"run": axis(0), "running": axis(1), "ran": axis(2)})
    run, running, ran = [store.resolve(word) for word in ("run", "running", "ran")]
    assert all(item.resolution == "exact" for item in (run, running, ran))
    assert run.rule_keys & running.rule_keys & ran.rule_keys == {"run"}
    # Rule identities also exist without a loaded embedding for the inflection.
    assert store.rule_keys("walked") & store.rule_keys("walk") == {"walk"}


@pytest.mark.parametrize("blocked", [("poison",), ("poisoned",)])
def test_denylist_applies_to_whole_family_even_for_exact_embeddings(blocked):
    store = SyntheticEmbeddings({"poison": axis(0), "poisoned": axis(1), "apple": axis(2)}, blocked=blocked)
    for word in ("poison", "Poisoned", "poisons"):
        with pytest.raises(WordValidationError) as caught:
            store.resolve(word)
        assert caught.value.code == "blocked_word"


def test_resolution_and_morphology_are_cached(monkeypatch):
    calls = []
    original = embedding_module.morphology_rule_keys

    def count(token):
        calls.append(token)
        return original(token)

    monkeypatch.setattr(embedding_module, "morphology_rule_keys", count)
    store = SyntheticEmbeddings({"run": axis(0)})
    first = store.resolve("Running")
    assert store.resolve(" Running ") is first
    assert store.resolve("running").resolved_word == "run"
    assert calls == ["running"]


def test_missing_nlp_is_actionable_and_gameplay_cannot_skip_family_checks(monkeypatch):
    def unavailable(_word):
        raise MorphologyUnavailableError("Run backend/scripts/prepare_nlp.py")

    monkeypatch.setattr(embedding_module, "morphology_rule_keys", unavailable)
    store = SyntheticEmbeddings({"prepare": axis(0), "prepared": axis(1)})
    # The unchanged mathematical engine can still inspect its exact vectors.
    np.testing.assert_array_equal(store.lookup("prepared"), axis(1))
    for word in ("prepare", "prepared", "preparing"):
        with pytest.raises(WordValidationError, match="prepare_nlp.py") as caught:
            store.resolve(word)
        assert caught.value.code == "normalization_unavailable"


def test_native_store_resolves_full_model_before_fallback_without_changing_catalog(tmp_path):
    from gensim.models import KeyedVectors

    words = ["token" + chr(97 + index) for index in range(20)]
    model = KeyedVectors(vector_size=300)
    model.add_vectors(words + ["prepare", "prepared", "Apple", "apple"],
                      np.stack([axis(index) for index in range(24)]))
    path, vocabulary = tmp_path / "fixture.kv", tmp_path / "words.txt"
    model.save(str(path))
    vocabulary.write_text("\n".join(words))
    store = EmbeddingStore(path, vocabulary).load()
    fingerprint, matrix = store.fingerprint, store.matrix.copy()
    assert store.resolve("prepared").resolution == "exact"
    np.testing.assert_array_equal(store.resolve("prepared").vector, axis(21))
    np.testing.assert_array_equal(store.resolve("Apple").vector, axis(22))
    assert store.resolve("preparing").resolved_word == "prepare"
    assert store.fingerprint == fingerprint
    np.testing.assert_array_equal(store.matrix, matrix)

    # In a limited model an unavailable exact form may only use a loaded lemma.
    limited = EmbeddingStore(path, vocabulary, limit=21).load()
    assert limited.resolve("prepared").resolved_word == "prepare"
    np.testing.assert_array_equal(limited.resolve("prepared").vector, axis(20))
