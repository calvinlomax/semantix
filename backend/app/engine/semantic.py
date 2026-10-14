"""Bounded, lazy cosine rankings over validated prepared-model entries."""
from functools import cached_property, lru_cache

import numpy as np

from .embeddings import WordValidationError


class SemanticIndex:
    def __init__(self, store):
        self.store = store

    @cached_property
    def entries(self):
        self.store.load()
        model = getattr(self.store, '_model', None)
        source = model.index_to_key if model is not None else self.store.words
        words, indices = [], []
        for index, word in enumerate(source):
            try:
                self.store.resolve(word)
            except WordValidationError:
                continue
            words.append(word)
            indices.append(index)
        raw = model.vectors if model is not None else self.store.matrix
        # Keep the vectors memory mapped; only norms and scores are resident.
        norms = np.sqrt(np.einsum('ij,ij->i', raw, raw))
        return tuple(words), np.asarray(indices, dtype=int), raw, norms

    @cached_property
    def word_indices(self):
        return {word: index for index, word in enumerate(self.entries[0])}

    @lru_cache(maxsize=32)
    def ranking(self, target):
        words, indices, raw, norms = self.entries
        scores = (raw @ self.store.lookup(target)) / np.maximum(norms, 1e-12)
        scores = scores[indices]
        if not len(scores):
            raise WordValidationError("No valid guess embeddings are available.")
        order = np.argsort(-scores, kind='stable')
        ranks = np.empty(len(words), dtype=int)
        # Equal scores share a competition rank.
        sorted_scores = scores[order]
        starts = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
        ranks[order] = np.maximum.accumulate(np.where(starts, np.arange(len(words)) + 1, 0))
        return scores, order, ranks

    @cached_property
    def hint_words(self):
        return frozenset(self.store.hint_words)

    def neighbors(self, target, limit=100):
        words = self.entries[0]
        scores, order, _ = self.ranking(target)
        family = self.store.rule_keys(target)
        result = []
        for i in order:
            word = words[i]
            if word in self.hint_words and not self.store.rule_keys(word) & family:
                result.append((word, float(scores[i])))
                if len(result) >= limit:
                    break
        return result

    def feedback(self, target, resolved, history, used_keys):
        scores, _, ranks = self.ranking(target)
        rank = int(ranks[self.word_indices[resolved.resolved_word]])
        percentile = 100 * (len(scores) - rank) / max(1, len(scores) - 1)
        bucket = ('top 100' if rank <= 100 else 'burning' if percentile >= 99 else
                  'hot' if percentile >= 95 else 'warm' if percentile >= 80 else
                  'cold' if percentile >= 50 else 'ice cold')
        score = float(resolved.vector @ self.store.lookup(target))
        def direction(previous):
            return 'warmer' if score > previous + 1e-6 else 'colder' if score < previous - 1e-6 else 'same'
        prior = history[-1] if history else None
        pair = float(resolved.vector @ self.store.lookup(prior['resolved_word'])) if prior else None
        return dict(rank=rank, vocabulary_size=len(scores), percentile=percentile, bucket=bucket,
                    versus_best=direction(max(x['word_similarity'] for x in history)) if history else 'first',
                    versus_previous=direction(prior['word_similarity']) if prior else 'first',
                    same_family=bool(resolved.rule_keys & used_keys),
                    cluster=('same cluster' if pair >= .4 else 'new semantic cluster') if pair is not None else 'first',
                    large_semantic_jump=pair is not None and pair < .15,
                    previous_guess_similarity=pair)

    @lru_cache(maxsize=512)
    def target_quality(self, target):
        scores, order, _ = self.ranking(target)
        words = self.entries[0]
        nearest = [int(i) for i in order if words[i] != target][:100]
        family = self.store.rule_keys(target)
        duplicates = sum(bool(self.store.rule_keys(words[i]) & family) for i in nearest[:20])
        safe = self.neighbors(target, 100)
        close = sum(score >= .25 for _, score in safe)
        density = float(np.mean(scores[nearest])) if nearest else 0.
        reasons = []
        if target not in self.hint_words:
            reasons.append('not familiar enough for the safe vocabulary')
        if close < 3:
            reasons.append('too few close safe neighbors')
        if duplicates > 8:
            reasons.append('morphological duplicates dominate')
        if density < .15:
            reasons.append('poor semantic connectivity')
        safe_count = sum(words[i] in self.hint_words for i in nearest)
        if safe_count < min(5, len(nearest)):
            reasons.append('neighborhood dominated by non-display words')
        return dict(accepted=not reasons, reasons=reasons, close_safe_neighbors=close,
                    safe_neighbors=safe_count, family_duplicates=duplicates, density=density,
                    neighbor_gap=float(scores[nearest[0]] - scores[nearest[-1]]) if nearest else 0.)
