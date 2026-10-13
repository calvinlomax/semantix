"""Vocabulary sources are broad for guesses, strict and stable for puzzles."""
import hashlib
import json
from pathlib import Path

import pytest

from backend.app.engine.morphology import TOKEN
from backend.scripts.build_large_vocab import build_vocabularies, read_words

ROOT = Path(__file__).resolve().parents[2] / "data/vocabulary"


def test_builder_is_deterministic_filters_names_and_blocked_families_and_keeps_targets():
    lexical = {"river", "tree", "garden", "poison", "computer"}
    forms = lambda word: frozenset({"trees": "tree", "poisoned": "poison"}.get(word, word).split())
    frequencies = {"computer": 6, "garden": 5, "trees": 4}
    candidates = ["computer", "trees", "poison", "poisoned", "london", "Tree", "two words", "x", "it's", "abc3", "madeupzz"]
    args = dict(curated=["tree", "river", "tree"], lexical_words=lexical, proper_only={"london"},
                blocked=["poison"], size=5, frequency=lambda word: frequencies.get(word, 0), forms=forms)
    first = build_vocabularies(frequency_words=candidates, **args)
    second = build_vocabularies(frequency_words=reversed(candidates), **args)
    assert first == second
    guesses, targets, _ = first
    assert targets == ["tree", "river"]
    assert guesses == ["computer", "garden", "river", "tree", "trees"]


def test_builder_never_silently_drops_level_words_to_fit_budget():
    with pytest.raises(ValueError, match="every curated"):
        build_vocabularies(["tree", "river"], [], set(), set(), [], size=1,
                           frequency=lambda _: 0, forms=lambda _: frozenset())


def test_committed_vocabularies_are_large_clean_reproducible_and_include_all_targets():
    guesses, targets = read_words(ROOT / "allowed_guesses.txt"), read_words(ROOT / "level_words.txt")
    old = read_words(ROOT / "words.txt")
    assert 50_000 <= len(guesses) <= 100_000
    assert len(guesses) > len(old) * 20
    assert guesses == sorted(set(guesses))
    assert 2000 <= len(targets) <= 10_000
    assert targets == old  # Preserves the calibrated target matrix order.
    assert set(targets) <= set(guesses)
    assert all(TOKEN.fullmatch(word) for word in guesses)
    assert not set(read_words(ROOT / "blocked.txt")) & set(guesses)
    assert not {"fucks", "fucking", "damned", "bitches", "london", "microsoft", "einstein", "washington"} & set(guesses)
    assert {"computer", "beautiful", "running", "river", "school", "music", "science", "garden", "painting", "memory"} <= set(guesses)
    manifest = json.loads((ROOT / "build_manifest.json").read_text())
    for filename, digest in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
    assert manifest["allowed_guess_count"] == len(guesses)
    assert manifest["level_word_count"] == len(targets)
