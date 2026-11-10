#INFO: Drawing helpers for the substrate, shared by both models


import numpy as np

from src.basic_fk.lattice import triangular_patch
from src.window import in_window


def substrate_cells(window, a_s=1.0):
    """
    Hexagonal substrate cells covering the window: centers at the energy minima, corners at the energy maxima.

    Parameters
    ----------
    window: Tuple (x_min, x_max, y_min, y_max)
    a_s: Spacing between neighboring substrate minima

    Returns
    ----------
    hexagons: List of arrays of shape (7, 2), the corners of each hexagon in order (first corner repeated to close it)
    """
    x_min, x_max, y_min, y_max = window

    #INFO: The minima are the triangular lattice with spacing a_s through the origin; triangular_patch builds exactly that lattice,
    # so build a patch big enough to cover the window and shift it by whole lattice vectors (an even number of rows keeps the row offsets right)
    i0 = int(np.floor(x_min / a_s)) - 2 # leftmost column needed
    j0 = 2 * (int(np.floor(y_min / (np.sqrt(3) * a_s))) - 1) # lowest row needed, rounded down to an even row
    nx = int(np.ceil((x_max - x_min) / a_s)) + 5
    ny = int(np.ceil((y_max - y_min) / (np.sqrt(3) / 2 * a_s))) + 6
    minima = triangular_patch(nx, ny, a_s) + np.array([i0 * a_s, j0 * (np.sqrt(3) / 2) * a_s])
    minima = minima[in_window(minima, window, pad=a_s)]

    # WARN: hexagon geometry. The maxima are at the triangle centers, a distance a_s / sqrt(3) from a minimum at angles 30, 90, ..., 330 degrees
    # (e.g. (a_s/2, a_s/(2 sqrt(3))) from the origin); the cell walls join neighboring maxima and pass through the saddles.
    angles = np.deg2rad(30 + 60 * np.arange(7)) # 7 angles so the last corner closes the hexagon
    corner_x = (a_s / np.sqrt(3)) * np.cos(angles)
    corner_y = (a_s / np.sqrt(3)) * np.sin(angles)
    # WARN: end

    hexagons = [np.column_stack([m[0] + corner_x, m[1] + corner_y]) for m in minima]
    return hexagons
