"""The display projection is stable and cannot affect 300D movement."""

import numpy as np
import pytest

from backend.app.engine.projection import Projection
from backend.app.engine.physics import step


@pytest.fixture
def anchors():
    values = np.random.default_rng(4).normal(size=(20, 300)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_three_axes_share_the_existing_two_dimensional_frame(anchors):
    projection = Projection.fit(anchors)
    assert projection.components.shape == (3, 300)
    assert projection.project(anchors[0]) == projection.project3d(anchors[0])[:2]
    assert len(projection.project3d(anchors[0])) == 3
    np.testing.assert_allclose(projection.components @ projection.components.T, np.eye(3), atol=1e-6)
    _, _, original_basis = np.linalg.svd(anchors - anchors.mean(axis=0), full_matrices=False)
    for axis in original_basis[:2]:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    np.testing.assert_array_equal(projection.components[:2], original_basis[:2])


def test_three_dimensional_projection_remains_fixed_after_movement(anchors):
    projection = Projection.fit(anchors)
    initial = projection.project_many3d(anchors)
    expected_position, expected_velocity = step(anchors[0], np.zeros(300), anchors[1])
    projection.project3d(expected_position)
    assert projection.project_many3d(anchors) == initial
    assert Projection.fit(anchors).project_many3d(anchors) == initial
    actual_position, actual_velocity = step(anchors[0], np.zeros(300), anchors[1])
    np.testing.assert_array_equal(actual_position, expected_position)
    np.testing.assert_array_equal(actual_velocity, expected_velocity)
    assert not projection.mean.flags.writeable
    assert not projection.components.flags.writeable


def test_small_anchor_sets_and_invalid_vectors():
    anchors = np.eye(300, dtype=np.float32)[:2]
    projection = Projection.fit(anchors)
    assert len(projection.project3d(anchors[0])) == 3
    assert projection.project3d(anchors[0])[2] == 0.0
    with pytest.raises(ValueError, match="finite 300-dimensional"):
        projection.project3d(np.full(300, np.nan))
    with pytest.raises(ValueError, match="finite 300-dimensional"):
        projection.project3d(np.zeros(3))
