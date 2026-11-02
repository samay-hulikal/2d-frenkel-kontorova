import matplotlib
matplotlib.use("Agg") # draw without opening windows
import matplotlib.pyplot as plt
import numpy as np

from src.lattice import triangular_patch
from src.plotting import central_window, in_window, particle_cells, plot_state, substrate_cells
from src.substrate import substrate_energy

V0, A_S = 0.7, 1.3 # non-unit values so wrong scalings show up


def test_window_has_the_right_area_and_center():
    pos0 = triangular_patch(20, 16, a=A_S)
    x_min, x_max, y_min, y_max = central_window(pos0, fraction=0.25)
    width = pos0[:, 0].max() - pos0[:, 0].min()
    height = pos0[:, 1].max() - pos0[:, 1].min()
    assert np.isclose((x_max - x_min) * (y_max - y_min), 0.25 * width * height)
    assert np.isclose((x_min + x_max) / 2, (pos0[:, 0].min() + pos0[:, 0].max()) / 2)
    assert np.isclose((y_min + y_max) / 2, (pos0[:, 1].min() + pos0[:, 1].max()) / 2)


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


def test_commensurate_particle_cells_match_substrate_cells():
    pos = triangular_patch(20, 16, a=A_S)
    window = central_window(pos)
    cells, which = particle_cells(pos, window)
    area = [0.5 * abs(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])) for c in cells]
    assert np.allclose(area, (np.sqrt(3) / 2) * A_S**2) # same area as a substrate hexagon
    assert np.all(in_window(pos[which], window, pad=2.0))


def test_plot_state_builds_two_panels():
    pos = triangular_patch(20, 16, a=A_S)
    fig = plot_state(pos, np.zeros(pos.shape[0]), central_window(pos), A_S, title="test")
    assert len(fig.axes) == 3 # two panels and the colorbar
    plt.close(fig)
