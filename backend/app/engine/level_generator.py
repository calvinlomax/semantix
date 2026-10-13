"""Bounded search and empirical calibration over the active Google embeddings.

Thresholds are measured from actual simulated trajectories. A saved solution is
always replayed with the authoritative physics before a level is accepted. The
search is a feasibility certificate, not a proof of optimality or human difficulty.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .embeddings import EmbeddingStore, SyntheticEmbeddings
from .physics import PhysicsConfig, is_forbidden, step
from .scoring import DIMENSIONS, cosine_similarity, normalize


@dataclass(frozen=True)
class Level:
    id: str
    difficulty: str
    start_word: str
    target_word: str
    move_limit: int
    threshold: float
    solution: tuple[str, ...]
    initial_similarity: float
    feedback_bands: tuple[float, ...]
    landmark_words: tuple[str, ...]
    model_fingerprint: str
    mechanics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Level":
        data = dict(value)
        for key in ("solution", "feedback_bands", "landmark_words"):
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass
class SearchNode:
    position: np.ndarray
    velocity: np.ndarray
    path: tuple[str, ...]
    similarities: tuple[float, ...]
    score: float
    previous_word: np.ndarray


@dataclass
class SearchResult:
    best_by_depth: list[SearchNode]
    beams: list[list[SearchNode]]


class LevelGenerationError(RuntimeError):
    pass


class LevelGenerator:
    MOVE_LIMITS = {"easy": 8, "medium": 6, "hard": 5}

    def __init__(
        self,
        store: EmbeddingStore | SyntheticEmbeddings,
        config: PhysicsConfig = PhysicsConfig(),
        *,
        beam_width: int = 6,
        candidate_count: int = 32,
    ) -> None:
        if beam_width < 1 or candidate_count < 2:
            raise ValueError("Search needs a positive beam width and at least two candidates.")
        self.store = store
        self.config = config
        self.beam_width = beam_width
        self.candidate_count = candidate_count

    def _candidates(self, position: np.ndarray, target: np.ndarray, target_word: str) -> np.ndarray:
        words, matrix = self.store.words, self.store.matrix
        if len(words) <= self.candidate_count:
            indices = np.arange(len(words))
        else:
            # Target neighbors provide sensible words; tangent neighbors provide
            # course corrections when residual momentum points past the target.
            tangent = target - float(position @ target) * position
            half = self.candidate_count // 2
            close = np.argsort(-(matrix @ target), kind="stable")[:half]
            steer = np.argsort(-(matrix @ tangent), kind="stable")[:self.candidate_count - half]
            indices = np.unique(np.concatenate((close, steer)))
        return np.asarray([index for index in indices if words[int(index)] != target_word], dtype=int)

    def _expand(self, node: SearchNode, target: np.ndarray, target_word: str,
                mechanics: dict[str, Any] | None = None) -> list[SearchNode]:
        mechanics = mechanics or {}
        changes = {key: mechanics[key] for key in ("gravity_strength", "power_multiplier", "forbidden_similarity") if key in mechanics}
        config = PhysicsConfig(**{**asdict(self.config), **changes}) if changes else self.config
        indices = self._candidates(node.position, target, target_word)
        if mechanics.get("forbidden_words") or mechanics.get("forbidden_regions"):
            regions = [self.store.lookup(word) for word in mechanics.get("forbidden_regions", [])]
            indices = np.asarray([index for index in indices if not is_forbidden(
                self.store.words[int(index)], self.store.matrix[index],
                forbidden_words=mechanics.get("forbidden_words", []), forbidden_vectors=regions,
                threshold=config.forbidden_similarity,
            )], dtype=int)
        if not len(indices):
            return []
        candidates = self.store.matrix[indices]
        origin = node.previous_word if config.movement_model == "displacement" else normalize(node.position)
        directions = candidates - origin
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms > 1e-12)
        velocities = config.momentum * node.velocity + config.impulse * directions
        if mechanics.get("attractor") and config.gravity_strength:
            velocities = velocities + config.gravity_strength * normalize(
                self.store.lookup(mechanics["attractor"]) - normalize(node.position), allow_zero=True
            )
        positions = normalize(node.position) + velocities
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        positions = np.divide(positions, norms, out=np.broadcast_to(node.position, positions.shape).copy(), where=norms > 1e-12)
        scores = np.clip(positions @ target, -1.0, 1.0)
        order = np.argsort(-scores, kind="stable")[:self.beam_width]
        return [SearchNode(
            positions[index], velocities[index],
            node.path + (self.store.words[int(indices[index])],),
            node.similarities + (float(scores[index]),), float(scores[index]), candidates[index],
        ) for index in order]

    def search(
        self,
        start_word: str,
        target_word: str,
        move_limit: int,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        previous_word: np.ndarray | None = None,
        mechanics: dict[str, Any] | None = None,
    ) -> SearchResult:
        target = self.store.lookup(target_word)
        start = self.store.lookup(start_word) if position is None else normalize(position)
        velocity = np.zeros(DIMENSIONS, dtype=np.float32) if velocity is None else velocity
        previous_word = self.store.lookup(start_word) if previous_word is None else previous_word
        initial = SearchNode(start, velocity, (), (), cosine_similarity(start, target), previous_word)
        beam = [initial]
        best, beams = [], []
        for _ in range(move_limit):
            expanded = [child for node in beam for child in self._expand(node, target, target_word, mechanics)]
            if not expanded:
                break
            expanded.sort(key=lambda node: (-node.score, node.path))
            # Deduplicate near-identical states so synonymous paths do not crowd
            # every useful alternative out of the bounded beam.
            unique: dict[bytes, SearchNode] = {}
            for node in expanded:
                signature = np.round(np.concatenate((node.position, node.velocity)), 4).tobytes()
                if signature not in unique:
                    unique[signature] = node
                if len(unique) >= self.beam_width:
                    break
            beam = list(unique.values())
            best.append(beam[0])
            beams.append(beam)
        return SearchResult(best, beams)

    def hint(
        self,
        level: Level,
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        previous_word: np.ndarray | None = None,
    ) -> str:
        result = self.search(level.start_word, level.target_word, 2, position=position,
                             velocity=velocity, previous_word=previous_word, mechanics=level.mechanics)
        if not result.best_by_depth:
            raise LevelGenerationError("No valid hint is available for this position.")
        # Look ahead one additional move to account for momentum.
        return result.best_by_depth[-1].path[0]

    def validate(self, level: Level) -> bool:
        if level.start_word not in self.store.words or level.target_word not in self.store.words:
            return False
        if level.model_fingerprint != self.store.fingerprint:
            return False
        if level.start_word == level.target_word or level.difficulty not in self.MOVE_LIMITS:
            return False
        if level.move_limit != self.MOVE_LIMITS[level.difficulty]:
            return False
        if not -1 < level.threshold <= 1 or not 1 <= len(level.solution) <= level.move_limit:
            return False
        if not np.isfinite(level.initial_similarity) or len(level.feedback_bands) != 4:
            return False
        if not np.isfinite(level.feedback_bands).all() or tuple(sorted(level.feedback_bands)) != tuple(level.feedback_bands):
            return False
        if any(value < -1 or value > level.threshold for value in level.feedback_bands):
            return False
        try:
            position = self.store.lookup(level.start_word)
            target = self.store.lookup(level.target_word)
            initial = cosine_similarity(position, target)
            if initial >= level.threshold or abs(initial - level.initial_similarity) > 1e-5:
                return False
            velocity = np.zeros(DIMENSIONS, dtype=np.float32)
            previous = position
            mechanics = level.mechanics or {}
            changes = {key: mechanics[key] for key in ("gravity_strength", "power_multiplier", "forbidden_similarity") if key in mechanics}
            config = PhysicsConfig(**{**asdict(self.config), **changes})
            attractor = self.store.lookup(mechanics["attractor"]) if mechanics.get("attractor") else None
            forbidden_vectors = [self.store.lookup(word) for word in mechanics.get("forbidden_regions", [])]
            for word in level.solution:
                if word == level.target_word:
                    return False
                vector = self.store.lookup(word)
                if is_forbidden(word, vector, forbidden_words=mechanics.get("forbidden_words", []),
                                forbidden_vectors=forbidden_vectors, threshold=config.forbidden_similarity):
                    return False
                position, velocity = step(position, velocity, vector, config, previous_word=previous, attractor=attractor)
                previous = vector
            return cosine_similarity(position, target) >= level.threshold
        except (ValueError, KeyError):
            return False

    def _landmarks(self, start: str, target: str) -> tuple[str, ...]:
        matrix, words = self.store.matrix, self.store.words
        start_sim = matrix @ self.store.lookup(start)
        target_sim = matrix @ self.store.lookup(target)
        ranked = np.concatenate((np.argsort(-start_sim, kind="stable")[:8],
                                 np.argsort(-target_sim, kind="stable")[:8],
                                 np.argsort(-(start_sim + target_sim), kind="stable")[:8]))
        return tuple(dict.fromkeys(words[int(index)] for index in ranked if words[int(index)] not in {start, target}))[:18]

    def generate(self, count_per_difficulty: int = 3, seed: int = 42) -> list[Level]:
        if count_per_difficulty < 1:
            raise ValueError("Request at least one level per difficulty.")
        words, matrix = self.store.words, self.store.matrix
        if len(words) < 20:
            raise LevelGenerationError("Level calibration requires at least 20 curated words.")
        rng = np.random.default_rng(seed)
        pairs = rng.integers(0, len(words), size=(max(2048, len(words)), 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        sampled = np.sum(matrix[pairs[:, 0]] * matrix[pairs[:, 1]], axis=1)
        edges = np.quantile(sampled, [0.02, 0.40, 0.75, 0.98])
        # Random-pair upper quartiles are still weakly related in Word2Vec.
        # Easy starts instead come from the target's actual nearest neighbors.
        anchor_indices = rng.choice(len(words), size=min(128, len(words)), replace=False)
        neighbor_scores = []
        for index in anchor_indices:
            similarities = matrix @ matrix[index]
            nearest = np.argsort(-similarities, kind="stable")
            nearest = nearest[nearest != index][1:min(24, len(words) - 1)]
            neighbor_scores.extend(similarities[nearest].tolist())
        easy_edges = np.quantile(neighbor_scores, [0.10, 0.90])
        bands = {"easy": (float(easy_edges[0]), float(easy_edges[1])),
                 "medium": (float(edges[2]), float(edges[3])),
                 "hard": (float(edges[0]), float(edges[1]))}
        pools: dict[str, list[tuple[str, str, float, SearchResult]]] = {}
        for difficulty, (low, high) in bands.items():
            # Four-move certificates occupy narrower attainable-score intervals;
            # sample more pairs instead of relaxing the hard-mode requirements.
            pool_size = max(32, count_per_difficulty * 12) if difficulty == "hard" else max(12, count_per_difficulty * 4)
            pool = []
            used: set[tuple[str, str]] = set()
            for _ in range(pool_size * 16):
                target_index = int(rng.integers(len(words)))
                similarities = matrix @ matrix[target_index]
                if difficulty == "easy":
                    nearest = np.argsort(-similarities, kind="stable")
                    nearest = nearest[nearest != target_index][1:min(24, len(words) - 1)]
                    candidates = nearest[(similarities[nearest] >= low) & (similarities[nearest] <= high)]
                else:
                    candidates = np.flatnonzero((similarities >= low) & (similarities <= high))
                    candidates = candidates[candidates != target_index]
                if not len(candidates):
                    continue
                start_index = int(rng.choice(candidates))
                start, target = words[start_index], words[target_index]
                if (start, target) in used:
                    continue
                used.add((start, target))
                result = self.search(start, target, self.MOVE_LIMITS[difficulty])
                if len(result.best_by_depth) == self.MOVE_LIMITS[difficulty]:
                    pool.append((start, target, float(similarities[start_index]), result))
                if len(pool) >= pool_size:
                    break
            if len(pool) < count_per_difficulty:
                raise LevelGenerationError(f"Not enough {difficulty} pairs in the active vocabulary to calibrate levels.")
            pools[difficulty] = pool

        # These are observed scores, not hard-coded cosine cutoffs. Later
        # difficulties require a strictly smaller target neighborhood.
        thresholds: dict[str, float] = {}
        previous_threshold = -1.0
        for difficulty, depth in (("easy", 2), ("medium", 3), ("hard", 4)):
            observed = [entry[3].best_by_depth[depth - 1].score for entry in pools[difficulty]]
            measured = float(np.quantile(observed, 0.5))
            # The separation margin is a fraction of observed search progress.
            gains = [max(0.0, entry[3].best_by_depth[-1].score - entry[2]) for entry in pools[difficulty]]
            margin = max(0.002, float(np.median(gains)) * 0.025)
            floor = previous_threshold + margin
            minimum = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
            intervals = []
            for entry in pools[difficulty]:
                scores = [node.score for node in entry[3].best_by_depth]
                lower = max(entry[2], max(scores[:minimum - 1]), floor) + 2e-5
                upper = max(scores[minimum - 1:]) - 2e-5
                if upper > lower:
                    intervals.append((lower, upper))
            thresholds_to_try = [max(measured, floor)] + [(low + high) / 2 for low, high in intervals]
            # Choose the median-derived neighborhood when enough certificates
            # support it; otherwise select an observed attainable interval with
            # enough trajectories, preserving increasing difficulty thresholds.
            def coverage(value: float) -> int:
                return sum(low <= value <= high for low, high in intervals)
            thresholds[difficulty] = min(thresholds_to_try, key=lambda value: (
                max(0, count_per_difficulty - coverage(value)), abs(value - measured), value,
            ))
            previous_threshold = thresholds[difficulty]

        levels: list[Level] = []
        for difficulty in self.MOVE_LIMITS:
            threshold = thresholds[difficulty]
            candidates = []
            for start, target, initial, result in pools[difficulty]:
                winning_depth = next((index for index, node in enumerate(result.best_by_depth, 1)
                                      if node.score >= threshold + 1e-5), None)
                # Reject one-shot solutions and levels solved before the intended
                # minimum number of search steps for this difficulty.
                minimum = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
                if winning_depth is None or winning_depth < minimum or initial >= threshold:
                    continue
                node = result.best_by_depth[winning_depth - 1]
                observed = np.asarray((initial,) + node.similarities, dtype=np.float32)
                feedback_bands = tuple(float(value) for value in np.quantile(observed, [0.15, 0.35, 0.55, 0.8]))
                identity = f"{self.store.fingerprint}|{seed}|{difficulty}|{start}|{target}|{threshold:.8f}|{asdict(self.config)}"
                alternatives = sum(candidate.score >= threshold for candidate in result.beams[winning_depth - 1])
                level = Level(
                    id=hashlib.sha256(identity.encode()).hexdigest()[:20], difficulty=difficulty,
                    start_word=start, target_word=target, move_limit=self.MOVE_LIMITS[difficulty],
                    threshold=threshold, solution=node.path, initial_similarity=initial,
                    feedback_bands=feedback_bands, landmark_words=self._landmarks(start, target),
                    model_fingerprint=self.store.fingerprint,
                    calibration={"method": "observed_bounded_beam_search", "seed": seed,
                                 "beam_width": self.beam_width, "candidate_count": self.candidate_count,
                                 "shortest_found": winning_depth, "alternatives_found": alternatives,
                                 "initial_similarity_band": list(bands[difficulty]),
                                 "solution_similarities": list(node.similarities),
                                 "thresholds": dict(thresholds), "sampled_pairs": len(pairs),
                                 "human_playtested": False},
                )
                if self.validate(level):
                    candidates.append(level)
            if len(candidates) < count_per_difficulty:
                raise LevelGenerationError(
                    f"Only {len(candidates)} validated {difficulty} levels met the measured threshold "
                    f"({threshold:.3f}); need {count_per_difficulty}. Increase the vocabulary or try a different seed."
                )
            # Prefer more alternative valid trajectories, then shorter solutions.
            candidates.sort(key=lambda level: (-level.calibration["alternatives_found"], len(level.solution), level.id))
            levels.extend(candidates[:count_per_difficulty])
        return levels
