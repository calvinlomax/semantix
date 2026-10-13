"""Lazy Google News loading plus a compact, normalized puzzle vocabulary."""

from __future__ import annotations

import hashlib
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from .scoring import DIMENSIONS, normalize

TOKEN = re.compile(r"^[a-z]{2,24}$")


class ModelUnavailableError(RuntimeError):
    pass


class WordValidationError(ValueError):
    def __init__(self, message: str, code: str = "unknown_word") -> None:
        super().__init__(message)
        self.code = code


def clean_word(word: str) -> str:
    cleaned = word.strip().lower()
    if not TOKEN.fullmatch(cleaned):
        raise WordValidationError("Use one English word of 2–24 letters.", "invalid_word")
    return cleaned


def _read_words(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(dict.fromkeys(
        line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))


class EmbeddingStore:
    def __init__(
        self,
        model_path: str | Path,
        vocabulary_path: str | Path,
        *,
        denylist_path: str | Path | None = None,
        binary: bool = False,
        limit: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.vocabulary_path = Path(vocabulary_path)
        self.denylist_path = Path(denylist_path) if denylist_path else None
        self.binary = binary
        self.limit = limit
        self._lock = threading.RLock()
        self._model = None
        self._state = "not_loaded"
        self._error: str | None = None
        self._words: tuple[str, ...] = ()
        self._matrix = np.empty((0, DIMENSIONS), dtype=np.float32)
        self._fingerprint = ""
        self._blocked: frozenset[str] = frozenset()
        self._indices: dict[str, int] = {}

    def status(self) -> dict:
        return {
            "state": self._state,
            "ready": self._state == "ready",
            "model": "word2vec-google-news-300",
            "dimensions": DIMENSIONS,
            "path": str(self.model_path),
            "exists": self.model_path.is_file(),
            "curated_words": len(self._words),
            "vocabulary_size": len(self._model) if self._model is not None else 0,
            "error": self._error,
            "fingerprint": self._fingerprint or None,
        }

    def load(self) -> "EmbeddingStore":
        if self._state == "ready":
            return self
        with self._lock:
            if self._state == "ready":
                return self
            self._state = "loading"
            try:
                if not self.model_path.is_file():
                    raise FileNotFoundError(
                        f"Google News model not found at {self.model_path}. "
                        "Run backend/scripts/prepare_model.py before starting a game."
                    )
                from gensim.models import KeyedVectors

                if self.binary or self.model_path.name.endswith((".bin", ".bin.gz")):
                    model = KeyedVectors.load_word2vec_format(
                        str(self.model_path), binary=True, limit=self.limit
                    )
                else:
                    model = KeyedVectors.load(str(self.model_path), mmap="r")
                if model.vector_size != DIMENSIONS:
                    raise ValueError(f"Expected the 300-dimensional Google News model; received {model.vector_size} dimensions.")
                self._blocked = frozenset(_read_words(self.denylist_path))
                words = [word for word in _read_words(self.vocabulary_path)
                         if TOKEN.fullmatch(word) and word not in self._blocked and word in model
                         and (self.limit is None or model.key_to_index[word] < self.limit)]
                if len(words) < 20:
                    raise ValueError("Fewer than 20 curated words are available in the model. Prepare a larger Google News vocabulary.")
                matrix = np.stack([normalize(model[word]) for word in words])
                matrix.flags.writeable = False
                fingerprint = hashlib.sha256()
                fingerprint.update("\n".join(words).encode())
                fingerprint.update(matrix.tobytes())
                self._model, self._words, self._matrix = model, tuple(words), matrix
                self._indices = {word: index for index, word in enumerate(words)}
                self._fingerprint = fingerprint.hexdigest()[:24]
                self._state, self._error = "ready", None
            except Exception as exc:
                self._state, self._error = "error", str(exc)
                raise ModelUnavailableError(self._error) from exc
        return self

    @property
    def words(self) -> tuple[str, ...]:
        self.load()
        return self._words

    @property
    def matrix(self) -> np.ndarray:
        self.load()
        return self._matrix

    @property
    def fingerprint(self) -> str:
        self.load()
        return self._fingerprint

    def lookup(self, word: str) -> np.ndarray:
        self.load()
        return self._lookup_normalized(clean_word(word))

    @lru_cache(maxsize=4096)
    def _lookup_normalized(self, word: str) -> np.ndarray:
        if word in self._blocked:
            raise WordValidationError("That word is unavailable in Semantix.", "blocked_word")
        if word in self._indices:
            return self._matrix[self._indices[word]]
        if word not in self._model or (self.limit is not None and self._model.key_to_index[word] >= self.limit):
            raise WordValidationError("That word isn't in the active Google News vocabulary. Try another word.")
        try:
            vector = normalize(self._model[word])
        except ValueError as exc:
            raise WordValidationError("That word has no usable embedding.") from exc
        vector.flags.writeable = False
        return vector


class SyntheticEmbeddings:
    """Explicit dependency-injected fixture. Never selected as an app fallback."""

    def __init__(self, vectors: Mapping[str, np.ndarray], *, blocked: tuple[str, ...] = ()) -> None:
        self._blocked = frozenset(blocked)
        self.words = tuple(sorted(word for word in vectors if word not in self._blocked))
        if not self.words:
            raise ValueError("At least one fixture vector is required.")
        self.matrix = np.stack([normalize(vectors[word]) for word in self.words])
        if self.matrix.shape[1] != DIMENSIONS:
            raise ValueError("Synthetic fixtures must also use 300 dimensions.")
        self.matrix.flags.writeable = False
        self._indices = {word: index for index, word in enumerate(self.words)}
        self.fingerprint = "synthetic-" + hashlib.sha256(self.matrix.tobytes()).hexdigest()[:16]

    def load(self) -> "SyntheticEmbeddings":
        return self

    def lookup(self, word: str) -> np.ndarray:
        word = clean_word(word)
        if word in self._blocked:
            raise WordValidationError("That word is unavailable in Semantix.", "blocked_word")
        if word not in self._indices:
            raise WordValidationError("Unknown fixture word.")
        return self.matrix[self._indices[word]]

    def status(self) -> dict:
        return {"state": "ready", "ready": True, "model": "synthetic-test-fixture", "dimensions": DIMENSIONS,
                "exists": True, "path": None, "curated_words": len(self.words), "vocabulary_size": len(self.words),
                "error": None, "fingerprint": self.fingerprint}
