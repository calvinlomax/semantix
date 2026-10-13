"""Cached English morphology, without downloads or NLP model loads per request.

WordNet provides dictionary-checked inflections, including irregular exceptions.
We retain all analyses rather than picking the shortest lemma: an isolated word
such as ``axes`` or ``better`` does not provide enough context to choose a sense.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path

TOKEN = re.compile(r"^[a-z]{2,24}$")
NLP_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "nltk_data"
_LOAD_LOCK = threading.RLock()


class MorphologyUnavailableError(RuntimeError):
    pass


def normalized_token(word: str) -> str:
    """Normalize case/Unicode presentation, never spelling or punctuation."""
    return unicodedata.normalize("NFKC", word.strip()).lower()


@lru_cache(maxsize=1)
def _lemmatizer():
    # NLTK's lazy corpus initialization mutates shared state, so initialize it
    # under one lock before concurrent requests perform dictionary-only reads.
    with _LOAD_LOCK:
        try:
            import nltk
            from nltk.corpus import wordnet
            from nltk.stem import WordNetLemmatizer

            location = Path(os.getenv("SEMANTIX_NLTK_DATA", str(NLP_DATA_PATH))).expanduser()
            if not location.is_absolute():
                location = NLP_DATA_PATH.parents[1] / location
            location = str(location)
            if location not in nltk.data.path:
                nltk.data.path.insert(0, location)
            wordnet.ensure_loaded()
            return WordNetLemmatizer()
        except (ImportError, LookupError, OSError) as exc:
            raise MorphologyUnavailableError(
                "Word recognition needs the local NLTK WordNet data. "
                "Run .venv/bin/python backend/scripts/prepare_nlp.py, then restart the backend."
            ) from exc


@lru_cache(maxsize=8192)
def lemma_forms(word: str) -> frozenset[str]:
    """All lexical analyses of a normalized token, without synonym expansion."""
    lemmatizer = _lemmatizer()
    return frozenset(
        lemma
        for part_of_speech in ("n", "v", "a", "r")
        # This is NLTK's documented all-lemma API. lemmatize() would choose the
        # shortest result and silently disambiguate forms like axes -> ax.
        for lemma in lemmatizer._morphy(word, part_of_speech)
        if TOKEN.fullmatch(lemma)
    )


def rule_keys(word: str) -> frozenset[str]:
    token = normalized_token(word)
    return frozenset({token}) | lemma_forms(token)
