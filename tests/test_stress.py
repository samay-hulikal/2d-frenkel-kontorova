import numpy as np

from src.lattice import bonds_nn, triangular_patch
from src.stress import isotropic_stress, voronoi_areas

K, A = 2.0, 1.3 # non-unit values so wrong scalings show up
NX, NY = 8, 7


def interior(bonds, N):
    """Particles with all 6 bonds, i.e. not on the outer ring."""
    counts = np.bincount(bonds.ravel(), minlength=N)
    return counts == 6


def test_interior_cells_are_regular_hexagons():
    L = 1.1
    pos = triangular_patch(NX, NY, a=L)
    bonds = bonds_nn(pos, a=L)
    areas = voronoi_areas(pos)
    inside = interior(bonds, pos.shape[0])
    assert np.allclose(areas[inside], (np.sqrt(3) / 2) * L**2)


def test_corners_of_patch_are_unbounded():
    pos = triangular_patch(NX, NY, a=A)
    areas = voronoi_areas(pos)
    assert np.isnan(areas[0]) and np.isnan(areas[-1])


def test_no_stress_at_rest_length():
    pos = triangular_patch(NX, NY, a=A)
    bonds = bonds_nn(pos, a=A)
    sigma = isotropic_stress(pos, bonds, voronoi_areas(pos), K, A)
    inside = interior(bonds, pos.shape[0])
    assert np.allclose(sigma[inside], 0.0)


def test_uniform_stretch_matches_hand_result():
    # Every bond has length L; interior particle: 6 bonds, area (sqrt(3)/2) L^2, so
    # sigma_iso = (1 / (2A)) * 6 * (1/2) k (L - a) L = sqrt(3) k (L - a) / L
    for L in (1.4, 1.2): # stretched, then compressed
        pos = triangular_patch(NX, NY, a=L)
        bonds = bonds_nn(pos, a=L)
        sigma = isotropic_stress(pos, bonds, voronoi_areas(pos), K, A)
        inside = interior(bonds, pos.shape[0])
        assert np.allclose(sigma[inside], np.sqrt(3) * K * (L - A) / L)
        assert np.all(np.sign(sigma[inside]) == np.sign(L - A)) # positive = expanded
