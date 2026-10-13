"""Small synthetic binary fixtures verify the real-file preparation pipeline."""
import gzip
import hashlib
import json

import numpy as np
import pytest
from gensim.models import KeyedVectors

from backend.scripts.prepare_model import extract_curated, prepare, read_vocabulary


def binary_fixture(path, *, dimension=300):
    vectors = {
        "river": np.arange(1, dimension + 1, dtype=np.float32) / 100,
        "forest": np.arange(dimension, 0, -1, dtype=np.float32) / 50,
        "Ocean": np.ones(dimension, dtype=np.float32),
    }
    data = f"{len(vectors)} {dimension}\n".encode()
    for word, vector in vectors.items():
        data += word.encode() + b" " + vector.astype("<f4").tobytes() + b"\n"
    path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)
    return vectors


@pytest.mark.parametrize("filename", ["fixture.bin", "fixture.gz"])
def test_stream_preserves_exact_vectors_and_vocabulary_order(tmp_path, filename):
    source = tmp_path / filename
    vectors = binary_fixture(source)
    words, matrix, detail = extract_curated(source, ["forest", "river", "ocean", "unavailable"])
    assert words == ["forest", "river"]
    np.testing.assert_array_equal(matrix[0], vectors["forest"])
    np.testing.assert_array_equal(matrix[1], vectors["river"])
    assert detail["missing_words"] == ["ocean", "unavailable"]
    assert detail["original_word_count"] == 3


def test_preparation_produces_native_mmap_and_manifest(tmp_path):
    source = tmp_path / "fixture.bin"
    vectors = binary_fixture(source)
    vocabulary = tmp_path / "words.txt"
    vocabulary.write_text("# comment\nriver\nforest\nriver\nmissing\n\n")
    assert read_vocabulary(vocabulary) == ["river", "forest", "missing"]
    output = tmp_path / "prepared.kv"
    manifest = prepare(source, output, vocabulary)
    saved = KeyedVectors.load(str(output), mmap="r")
    assert isinstance(saved.vectors, np.memmap)
    assert saved.vector_size == 300
    np.testing.assert_array_equal(saved["river"], vectors["river"])
    assert manifest["reduced_vocabulary"] is True
    assert manifest["requested_word_count"] == 3
    assert manifest["missing_words"] == ["missing"]
    assert manifest["missing_word_count"] == 1
    for name, detail in manifest["artifacts"].items():
        artifact = output.parent / name
        assert artifact.stat().st_size == detail["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == detail["sha256"]
    assert json.loads(output.with_suffix(".kv.manifest.json").read_text())["retained_word_count"] == 2


def test_preparation_rejects_wrong_dimensions_and_truncated_source(tmp_path):
    source = tmp_path / "fixture.bin"
    binary_fixture(source, dimension=10)
    with pytest.raises(ValueError, match="300-dimensional"):
        extract_curated(source, ["river"])
    source.write_bytes(b"1 300\nriver " + b"\0" * 100)
    with pytest.raises(ValueError, match="truncated"):
        extract_curated(source, ["river"])


def test_preparation_rejects_zero_vectors(tmp_path):
    source = tmp_path / "fixture.bin"
    source.write_bytes(b"1 300\nriver " + b"\0" * 1200)
    with pytest.raises(ValueError, match="None of the curated"):
        extract_curated(source, ["river"])


def test_default_preparation_streams_guess_source_and_never_loads_full_model(tmp_path, monkeypatch):
    from backend.scripts import prepare_model
    source = tmp_path / "fixture.bin"
    binary_fixture(source)
    vocabulary = tmp_path / "allowed_guesses.txt"
    vocabulary.write_text("river\nforest\n")
    monkeypatch.setattr(prepare_model, "DEFAULT_VOCABULARY", vocabulary)
    monkeypatch.setattr(KeyedVectors, "load_word2vec_format", lambda *_args, **_kwargs: pytest.fail("Full model loading is forbidden"))
    manifest = prepare(source, tmp_path / "reduced.kv")
    assert manifest["reduced_vocabulary"] and manifest["retained_word_count"] == 2
