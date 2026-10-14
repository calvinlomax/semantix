"""Lazy Google News loading plus a compact, normalized puzzle vocabulary."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from .scoring import DIMENSIONS, normalize
from .morphology import (
    TOKEN, MorphologyUnavailableError, lemma_forms, normalized_token,
    rule_keys as morphology_rule_keys,
)

MAX_PREPARED_WORDS = 150_000


class ModelUnavailableError(RuntimeError):
    pass


class WordValidationError(ValueError):
    def __init__(self, message: str, code: str = "unknown_word") -> None:
        super().__init__(message)
        self.code = code


def clean_word(word: str) -> str:
    cleaned = normalized_token(word)
    if not TOKEN.fullmatch(cleaned):
        raise WordValidationError("Use one English word of 2–24 letters.", "invalid_word")
    return cleaned


@dataclass(frozen=True, eq=False)
class WordResolution:
    original_word: str
    resolved_word: str
    resolution: str
    vector: np.ndarray
    rule_keys: frozenset[str]


class _WordResolver:
    """Shared authoritative resolution for real and injected fixture models."""

    @lru_cache(maxsize=8192)
    def rule_keys(self, word: str) -> frozenset[str]:
        token = clean_word(word)
        try:
            return morphology_rule_keys(token)
        except MorphologyUnavailableError as exc:
            raise WordValidationError(str(exc), "normalization_unavailable") from exc

    @cached_property
    def _blocked_keys(self) -> frozenset[str]:
        return frozenset(key for word in self._blocked for key in self.rule_keys(word))

    def resolve(self, word: str) -> WordResolution:
        self.load()
        clean_word(word)
        return self._resolve_original(word.strip())

    @lru_cache(maxsize=4096)
    def _resolve_original(self, original: str) -> WordResolution:
        token = clean_word(original)
        if token in self._blocked:
            raise WordValidationError("That word is unavailable in Semantix.", "blocked_word")
        keys = self.rule_keys(token)
        if keys & self._blocked_keys:
            raise WordValidationError("That word or its base form is unavailable in Semantix.", "blocked_word")

        if self._has_word(original):
            resolved, method = original, "exact"
        elif self._has_word(token):
            resolved, method = token, "normalized"
        else:
            candidates = lemma_forms(token) - {token}
            if len(candidates) > 1:
                raise WordValidationError(
                    f'"{original}" has more than one possible base form '
                    f'({", ".join(sorted(candidates))}). Enter the intended base word.',
                    "ambiguous_word",
                )
            if not candidates or not self._has_word(next(iter(candidates))):
                raise WordValidationError(
                    f'Neither "{original}" nor a recognized base form is in the active '
                    "Google News vocabulary. Try another word."
                )
            resolved, method = next(iter(candidates)), "lemma"
        return WordResolution(original, resolved, method, self._exact_vector(resolved), keys)

    def lookup(self, word: str) -> np.ndarray:
        """Vector access for catalog/math; gameplay must use resolve() for rules.

        Exact vectors remain usable for preparation/validation without WordNet.
        Resolution additionally requires morphology to enforce word families.
        """
        self.load()
        token = clean_word(word)
        if token in self._blocked:
            raise WordValidationError("That word is unavailable in Semantix.", "blocked_word")
        original = word.strip()
        if self._has_word(original):
            return self._exact_vector(original)
        if self._has_word(token):
            return self._exact_vector(token)
        return self.resolve(word).vector


def _read_words(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(dict.fromkeys(
        line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))


class EmbeddingStore(_WordResolver):
    def __init__(
        self,
        model_path: str | Path,
        vocabulary_path: str | Path,
        *,
        denylist_path: str | Path | None = None,
        target_vocabulary_path: str | Path | None = None,
        hint_vocabulary_path: str | Path | None = None,
        require_prepared: bool = False,
        binary: bool = False,
        limit: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.vocabulary_path = Path(vocabulary_path)
        # The small matrix is only for puzzle search and calibration.
        # Runtime guess membership comes from the prepared model itself.
        self.target_vocabulary_path = Path(target_vocabulary_path) if target_vocabulary_path else self.vocabulary_path
        self.hint_vocabulary_path = Path(hint_vocabulary_path) if hint_vocabulary_path else self.target_vocabulary_path
        self._hints = ()
        self.require_prepared = require_prepared
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
        self._manifest: dict = {}
        self._target_requested_count = 0

    def status(self) -> dict:
        return {
            "state": self._state,
            "ready": self._state == "ready",
            "model": "word2vec-google-news-300",
            "dimensions": DIMENSIONS,
            "path": str(self.model_path),
            "exists": self.model_path.is_file(),
            "curated_words": len(self._words),
            "target_vocabulary_size": len(self._words),
            "hint_vocabulary_size": len(self._hints),
            "requested_guess_count": self._manifest.get("requested_word_count"),
            "missing_guess_count": len(self._manifest.get("missing_words", [])),
            "requested_target_count": self._target_requested_count,
            "reduced": self._manifest.get("reduced_vocabulary"),
            "memory_mapped": isinstance(self._model.vectors, np.memmap) if self._model is not None else False,
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
                if self.binary or self.model_path.name.endswith((".bin", ".bin.gz")):
                    raise ValueError("Runtime requires a reduced native .kv model. Extract allowed_guesses.txt with backend.scripts.prepare_model; never load the original Google News binary here.")
                if self.model_path.suffix != ".kv":
                    raise ValueError("SEMANTIX_MODEL_PATH must point to a prepared .kv file.")
                manifest_path = self.model_path.with_suffix(".kv.manifest.json")
                manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
                if self.require_prepared:
                    if (manifest.get("reduced_vocabulary") is not True
                            or manifest.get("dimensions") != DIMENSIONS
                            or not 1 <= manifest.get("retained_word_count", 0) <= MAX_PREPARED_WORDS):
                        raise ValueError("A reduced model preparation manifest with at most 150,000 retained words is required. Run backend.scripts.prepare_model with allowed_guesses.txt.")
                from gensim.models import KeyedVectors

                model = KeyedVectors.load(str(self.model_path), mmap="r")
                if model.vector_size != DIMENSIONS:
                    raise ValueError(f"Expected the 300-dimensional Google News model; received {model.vector_size} dimensions.")
                if len(model) > MAX_PREPARED_WORDS:
                    raise ValueError("This model exceeds the 150,000-word reduced-model limit; prepare a smaller guess vocabulary.")
                if self.require_prepared and (len(model) != manifest["retained_word_count"] or not isinstance(model.vectors, np.memmap)):
                    raise ValueError("The prepared .kv, vector sidecar, and manifest must be deployed together and support memory mapping.")
                self._blocked = frozenset(_read_words(self.denylist_path))
                requested_targets = _read_words(self.target_vocabulary_path)
                words = [word for word in requested_targets
                         if TOKEN.fullmatch(word) and word not in self._blocked and word in model
                         and not self.rule_keys(word) & self._blocked_keys
                         and (self.limit is None or model.key_to_index[word] < self.limit)]
                if len(words) < 20:
                    raise ValueError("Fewer than 20 curated words are available in the model. Prepare a larger Google News vocabulary.")
                matrix = np.stack([normalize(model[word]) for word in words])
                matrix.flags.writeable = False
                fingerprint = hashlib.sha256()
                fingerprint.update("\n".join(words).encode())
                fingerprint.update(matrix.tobytes())
                self._model, self._words, self._matrix = model, tuple(words), matrix
                self._hints = tuple(word for word in _read_words(self.hint_vocabulary_path)
                                    if TOKEN.fullmatch(word) and word in model
                                    and not self.rule_keys(word) & self._blocked_keys)
                self._manifest = manifest
                self._target_requested_count = len(requested_targets)
                self._indices = {word: index for index, word in enumerate(words)}
                self._fingerprint = fingerprint.hexdigest()[:24]
                self._state, self._error = "ready", None
            except Exception as exc:
                self._state, self._error = "error", str(exc)
                raise ModelUnavailableError(self._error) from exc
        return self

    @property
    def hint_words(self):
        self.load()
        return self._hints

    @property
    def words(self) -> tuple[str, ...]:
        """Retained curated level words, never the broad accepted-guess pool."""
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

    def _has_word(self, word: str) -> bool:
        return word in self._model and (self.limit is None or self._model.key_to_index[word] < self.limit)

    @lru_cache(maxsize=4096)
    def _exact_vector(self, word: str) -> np.ndarray:
        if word in self._indices:
            return self._matrix[self._indices[word]]
        try:
            vector = normalize(self._model[word])
        except ValueError as exc:
            raise WordValidationError("That word has no usable embedding.") from exc
        vector.flags.writeable = False
        return vector


class SyntheticEmbeddings(_WordResolver):
    """Explicit dependency-injected fixture. Never selected as an app fallback."""

    def __init__(self, vectors: Mapping[str, np.ndarray], *, blocked: tuple[str, ...] = ()) -> None:
        self._blocked = frozenset(clean_word(word) for word in blocked)
        self.words = tuple(sorted(word for word in vectors if word not in self._blocked))
        if not self.words:
            raise ValueError("At least one fixture vector is required.")
        self.hint_words = self.words
        self.matrix = np.stack([normalize(vectors[word]) for word in self.words])
        if self.matrix.shape[1] != DIMENSIONS:
            raise ValueError("Synthetic fixtures must also use 300 dimensions.")
        self.matrix.flags.writeable = False
        self._indices = {word: index for index, word in enumerate(self.words)}
        self.fingerprint = "synthetic-" + hashlib.sha256(self.matrix.tobytes()).hexdigest()[:16]

    def load(self) -> "SyntheticEmbeddings":
        return self

    def _has_word(self, word: str) -> bool:
        return word in self._indices

    def _exact_vector(self, word: str) -> np.ndarray:
        return self.matrix[self._indices[word]]

    def status(self) -> dict:
        return {"state": "ready", "ready": True, "model": "synthetic-test-fixture", "dimensions": DIMENSIONS,
                "exists": True, "path": None, "curated_words": len(self.words), "vocabulary_size": len(self.words),
                "error": None, "fingerprint": self.fingerprint}
