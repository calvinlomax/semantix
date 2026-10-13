"""A stable PCA transformation fitted once and retained for a complete game."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scoring import DIMENSIONS


@dataclass(frozen=True)
class Projection:
    mean: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(cls, anchors: np.ndarray) -> "Projection":
        anchors = np.asarray(anchors, dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != DIMENSIONS or len(anchors) < 2:
            raise ValueError("PCA needs at least two 300-dimensional anchors.")
        if not np.isfinite(anchors).all():
            raise ValueError("PCA anchors must be finite.")
        mean = anchors.mean(axis=0)
        # Keep the first two axes identical to the existing 2D map. With only
        # two anchors, the missing third axis is zero rather than invented.
        components = np.zeros((3, DIMENSIONS), dtype=np.float32)
        _, _, basis = np.linalg.svd(anchors - mean, full_matrices=False)
        components[:min(3, len(basis))] = basis[:3]
        # SVD signs are arbitrary; fixing them prevents mirrored rounds.
        for component in components:
            if component[np.argmax(np.abs(component))] < 0:
                component *= -1
        mean.flags.writeable = False
        components.flags.writeable = False
        return cls(mean, components)

    def project(self, vector: np.ndarray) -> list[float]:
        """Project onto the first two axes for existing API consumers."""
        return self.project3d(vector)[:2]

    def project3d(self, vector: np.ndarray) -> list[float]:
        """Project with the same retained basis; never used for game physics."""
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (DIMENSIONS,) or not np.isfinite(value).all():
            raise ValueError("Projection requires a finite 300-dimensional vector.")
        return ((value - self.mean) @ self.components.T).astype(float).tolist()

    def project_many(self, vectors: np.ndarray) -> list[list[float]]:
        return [self.project(vector) for vector in vectors]

    def project_many3d(self, vectors: np.ndarray) -> list[list[float]]:
        return [self.project3d(vector) for vector in vectors]
