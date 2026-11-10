import matplotlib
matplotlib.use("Agg") # draw without opening windows
import matplotlib.pyplot as plt
import numpy as np

from src.basic_fk.lattice import triangular_patch
from src.viz.basic_fk import particle_cells, plot_state
from src.window import central_window, in_window

A_S = 1.3 # non-unit values so wrong scalings show up


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
