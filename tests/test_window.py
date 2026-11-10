import numpy as np

from src.basic_fk.lattice import triangular_patch
from src.window import central_window

A_S = 1.3 # non-unit value so wrong scalings show up


def test_window_has_the_right_area_and_center():
    pos0 = triangular_patch(20, 16, a=A_S)
    x_min, x_max, y_min, y_max = central_window(pos0, fraction=0.25)
    width = pos0[:, 0].max() - pos0[:, 0].min()
    height = pos0[:, 1].max() - pos0[:, 1].min()
    assert np.isclose((x_max - x_min) * (y_max - y_min), 0.25 * width * height)
    assert np.isclose((x_min + x_max) / 2, (pos0[:, 0].min() + pos0[:, 0].max()) / 2)
    assert np.isclose((y_min + y_max) / 2, (pos0[:, 1].min() + pos0[:, 1].max()) / 2)
