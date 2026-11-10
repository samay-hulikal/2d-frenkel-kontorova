import numpy as np

from src.substrate import substrate_energy
from src.viz.substrate import substrate_cells

V0, A_S = 0.7, 1.3 # non-unit values so wrong scalings show up


def test_substrate_hexagons_sit_on_minima_and_maxima():
    window = (3.1, 11.7, -2.4, 6.9) # arbitrary, including negative y
    hexagons = substrate_cells(window, A_S)
    centers = np.array([h[:6].mean(axis=0) for h in hexagons])
    corners = np.vstack([h[:6] for h in hexagons])
    assert np.allclose(substrate_energy(centers, V0, A_S), 0.0, atol=1e-12) # centers at minima
    assert np.allclose(substrate_energy(corners, V0, A_S), V0) # corners at maxima


def test_substrate_hexagons_cover_the_window():
    window = (3.1, 11.7, -2.4, 6.9)
    hexagons = substrate_cells(window, A_S)
    centers = np.array([h[:6].mean(axis=0) for h in hexagons])
    # every point of the window is within a_s / sqrt(3) (the hexagon's corner distance) of some center
    xs, ys = np.meshgrid(np.linspace(3.1, 11.7, 40), np.linspace(-2.4, 6.9, 40))
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    nearest = np.min(np.linalg.norm(grid[:, None, :] - centers[None, :, :], axis=2), axis=1)
    assert np.all(nearest <= A_S / np.sqrt(3) + 1e-9)
