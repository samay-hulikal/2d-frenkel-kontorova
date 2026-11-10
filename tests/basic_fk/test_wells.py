import numpy as np

from src.basic_fk.lattice import triangular_patch
from src.basic_fk.wells import fraction_left_well, nearest_minimum

A_S = 1.3 # non-unit value so wrong scalings show up


def test_nearest_minimum_matches_brute_force():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-9, 9, size=(2000, 2))
    minima = triangular_patch(40, 40, a=A_S) - np.array([20 * A_S, 20 * (np.sqrt(3) / 2) * A_S]) # covers the points, shifted by whole rows
    brute = minima[np.argmin(np.linalg.norm(pts[:, None, :] - minima[None, :, :], axis=2), axis=1)]
    ij = nearest_minimum(pts, A_S)
    labels_xy = ij[:, :1] * np.array([A_S, 0.0]) + ij[:, 1:] * np.array([A_S / 2, np.sqrt(3) / 2 * A_S])
    assert np.allclose(labels_xy, brute)


def test_minima_label_themselves():
    minima = triangular_patch(6, 5, a=A_S)
    labels_xy = nearest_minimum(minima, A_S) @ np.array([[A_S, 0.0], [A_S / 2, np.sqrt(3) / 2 * A_S]])
    assert np.allclose(labels_xy, minima)


def test_fraction_left_well():
    pos0 = triangular_patch(10, 10, a=A_S)
    pos = pos0.copy()
    pos[:10] += np.array([0.3 * A_S, 0.0]) # small shift: still inside the starting hexagon (half-width sqrt(3)/2 a_s / sqrt(3) = a_s/2)
    pos[10:20] += np.array([0.7 * A_S, 0.0]) # past the wall: now in the neighbor's well
    fraction, left = fraction_left_well(pos0, pos, A_S)
    assert np.isclose(fraction, 10 / 100)
    assert left[10:20].all() and not left[:10].any()
    which = np.zeros(100, dtype=bool)
    which[10:30] = True
    assert np.isclose(fraction_left_well(pos0, pos, A_S, which)[0], 0.5)
