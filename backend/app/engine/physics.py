"""Small, independently testable forces in the original 300-dimensional space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .scoring import DIMENSIONS, EPSILON, cosine_similarity, normalize


@dataclass(frozen=True)
class PhysicsConfig:
    momentum: float = 0.35
    impulse: float = 0.45
    movement_model: str = "direction"
    gravity_strength: float = 0.0
    power_multiplier: float = 1.5
    forbidden_similarity: float = 0.75

    def __post_init__(self) -> None:
        if not 0 <= self.momentum < 1:
            raise ValueError("Momentum must be in [0, 1).")
        if not 0 < self.impulse <= 2:
            raise ValueError("Impulse must be in (0, 2].")
        if self.movement_model not in {"direction", "displacement"}:
            raise ValueError("Movement model must be direction or displacement.")
        if not 0 <= self.gravity_strength <= 1:
            raise ValueError("Gravity strength must be in [0, 1].")
        if not 1 <= self.power_multiplier <= 3:
            raise ValueError("Power multiplier must be in [1, 3].")
        if not -1 <= self.forbidden_similarity <= 1:
            raise ValueError("Forbidden similarity must be in [-1, 1].")


def _vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (DIMENSIONS,) or not np.isfinite(value).all():
        raise ValueError(f"Gameplay requires finite {DIMENSIONS}-dimensional vectors.")
    return value


def step(
    position: np.ndarray,
    velocity: np.ndarray,
    word_vector: np.ndarray,
    config: PhysicsConfig = PhysicsConfig(),
    *,
    previous_word: np.ndarray | None = None,
    attractor: np.ndarray | None = None,
    power: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    position = normalize(_vector(position))
    velocity = _vector(velocity)
    word_vector = normalize(_vector(word_vector))
    origin = position
    if config.movement_model == "displacement" and previous_word is not None:
        origin = normalize(_vector(previous_word))
    direction = normalize(word_vector - origin, allow_zero=True)
    impulse = config.impulse * (config.power_multiplier if power else 1.0)
    next_velocity = config.momentum * velocity + impulse * direction
    if attractor is not None and config.gravity_strength:
        gravity = normalize(normalize(_vector(attractor)) - position, allow_zero=True)
        next_velocity = next_velocity + config.gravity_strength * gravity
    unnormalized = position + next_velocity
    # A perfectly cancelling force keeps the last valid position.
    next_position = position.copy() if np.linalg.norm(unnormalized) <= EPSILON else normalize(unnormalized)
    return next_position, np.asarray(next_velocity, dtype=np.float32)


def is_forbidden(
    word: str,
    vector: np.ndarray,
    *,
    forbidden_words: Iterable[str] = (),
    forbidden_vectors: Iterable[np.ndarray] = (),
    threshold: float = 0.75,
) -> bool:
    return word in forbidden_words or any(
        cosine_similarity(vector, forbidden) >= threshold for forbidden in forbidden_vectors
    )
