"""Scoring uses only original embedding coordinates, never map coordinates."""

from __future__ import annotations

import numpy as np

DIMENSIONS = 300
EPSILON = 1e-12


def normalize(vector: np.ndarray, *, allow_zero: bool = False) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Expected a finite one-dimensional vector.")
    norm = float(np.linalg.norm(value))
    if norm <= EPSILON:
        if allow_zero:
            return np.zeros_like(value)
        raise ValueError("Cannot normalize a zero-length vector.")
    return value / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))


def semantic_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def reached_target(similarity: float, threshold: float) -> bool:
    return similarity >= threshold


def feedback(similarity: float, threshold: float, bands: list[float] | tuple[float, ...]) -> str:
    """Bands are observed trajectory quantiles saved with each validated level."""
    if reached_target(similarity, threshold):
        return "Target reached"
    labels = ("Very cold", "Cold", "Warm", "Hot", "Very hot")
    index = sum(similarity >= boundary for boundary in bands[:4])
    return labels[index]


def efficiency(before: float, after: float, threshold: float) -> float:
    """Fraction of the remaining similarity gap closed, with regressions negative."""
    gap = threshold - before
    if gap <= EPSILON:
        return 0.0
    return float(np.clip((after - before) / gap, -1.0, 1.0))
