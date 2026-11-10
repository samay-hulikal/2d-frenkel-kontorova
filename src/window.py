#INFO: The central window: a rectangle in the middle of the patch where results are measured and drawn, away from edge effects


import numpy as np


def central_window(pos0, fraction=0.25):
    """
    Rectangle in the middle of the starting patch, fixed in space, covering a given fraction of its area.

    Parameters
    ----------
    pos0: Array such that pos0[k] gives the STARTING (x, y) coordinates of particle k
    fraction: Fraction of the patch's area inside the window (0.25 means half the width and half the height)

    Returns
    ----------
    window: Tuple (x_min, x_max, y_min, y_max)
    """
    x_lo, x_hi = pos0[:, 0].min(), pos0[:, 0].max()
    y_lo, y_hi = pos0[:, 1].min(), pos0[:, 1].max()
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2 # center of the patch
    s = np.sqrt(fraction) # each side shrinks by sqrt(fraction) so the area shrinks by fraction
    half_w = s * (x_hi - x_lo) / 2
    half_h = s * (y_hi - y_lo) / 2
    window = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
    return window


def in_window(pos, window, pad=0.0):
    """
    Which points lie inside the window, optionally enlarged by pad on every side.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of point k
    window: Tuple (x_min, x_max, y_min, y_max)
    pad: Extra margin added on every side

    Returns
    ----------
    inside: Boolean array, inside[k] is True if point k is inside the (padded) window
    """
    x_min, x_max, y_min, y_max = window
    x = pos[:, 0]
    y = pos[:, 1]
    inside = (x > x_min - pad) & (x < x_max + pad) & (y > y_min - pad) & (y < y_max + pad)
    return inside
