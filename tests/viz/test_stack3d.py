import numpy as np

from src.basic_fk.lattice import triangular_patch
from src.viz.stack3d import cell_mesh, layer_indices, plot_stack
from src.window import central_window


def test_layer_indices_evenly_spaced_and_distinct():
    times = np.arange(0, 101.0) # snapshots every 1
    assert np.array_equal(layer_indices(times, 5), [0, 25, 50, 75, 100])
    assert len(layer_indices(np.array([0.0, 1.0]), 8)) == 2 # never more layers than snapshots


def test_cell_mesh_triangle_fan_covers_a_hexagon():
    angles = np.deg2rad(60 * np.arange(6))
    hexagon = np.column_stack([np.cos(angles), np.sin(angles)])
    x, y, z, i, j, k, owner = cell_mesh([hexagon, hexagon + 3], z=2.0)
    assert len(i) == 2 * 4 and np.all(z == 2.0) and np.array_equal(np.unique(owner), [0, 1])
    area = 0.5 * np.abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
    assert np.isclose(area[owner == 0].sum(), 3 * np.sqrt(3) / 2) # unit hexagon area


def test_plot_stack_slider_and_button():
    pos = triangular_patch(16, 16, a=1.0)
    snaps = np.array([pos, pos * 1.01, pos * 1.02])
    left = np.zeros((3, pos.shape[0]), dtype=bool)
    left[2, :10] = True
    fig = plot_stack(np.array([0.0, 1.0, 2.0]), snaps, left, central_window(pos), n_layers=3)
    assert len(fig.data) == 6 # cells + walls per layer
    steps = fig.layout.sliders[0].steps
    assert len(steps) == 3 and list(steps[0].args[0]["visible"]) == [True, True, False, False, False, False]
    assert len(fig.layout.updatemenus[0].buttons) == 2
