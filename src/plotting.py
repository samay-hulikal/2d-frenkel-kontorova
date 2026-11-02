#INFO: Figures of the bees' lattice on the substrate; drawing only, the physics lives in the other modules


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import Voronoi

from src.lattice import triangular_patch

BLUE = "#1f5fbf" # substrate cells
YELLOW = "#b8860b" # bees' (particle) Voronoi cells, dark yellow


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


def particle_cells(pos, window, pad=2.0):
    """
    Voronoi cells of the particles near the window.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    window: Tuple (x_min, x_max, y_min, y_max)
    pad: Particles up to this far outside the window are included so cells touching the window border are drawn

    Returns
    ----------
    cells: List of arrays, cells[n] holds the corners (x, y) of the n-th cell in order
    which: Array of particle indices, which[n] is the particle that owns cells[n]
    """
    vor = Voronoi(pos)
    near = np.where(in_window(pos, window, pad))[0]

    cells = []
    which = []
    for k in near:
        region = vor.regions[vor.point_region[k]] # indices of the cell's corners, in order around the cell
        if len(region) == 0 or -1 in region:
            continue # unbounded cell on the outside of the patch; never inside a central window
        cells.append(vor.vertices[region])
        which.append(k)
    return cells, np.array(which, dtype=int)


def plot_state(pos, sigma_iso, window, a_s=1.0, title=None):
    """
    Two-panel figure of the window. Left: substrate cells (blue), particle cells (dark yellow), particles (black dots).
    Right: particle cells colored by isotropic stress (purple = compressed, green = expanded).

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    sigma_iso: Array such that sigma_iso[k] is the isotropic stress of particle k
    window: Tuple (x_min, x_max, y_min, y_max), the region shown
    a_s: Spacing between neighboring substrate minima
    title: Optional title for the whole figure

    Returns
    ----------
    fig: The matplotlib figure
    """
    hexagons = substrate_cells(window, a_s)
    cells, which = particle_cells(pos, window, pad=2 * a_s)

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")

    #INFO: Left panel; thick yellow underneath and thin blue on top, so both stay visible where they coincide
    left.add_collection(PolyCollection(cells, facecolors="none", edgecolors=YELLOW, linewidths=2.4, zorder=2))
    left.add_collection(LineCollection(hexagons, colors=BLUE, linewidths=0.8, zorder=3))
    dots = in_window(pos, window, pad=0.5 * a_s)
    left.plot(pos[dots, 0], pos[dots, 1], "k.", ms=3, zorder=4)
    left.set_title("Substrate cells (blue) and bee cells (yellow)")

    #INFO: Right panel; colors centered at zero and scaled by the largest |sigma_iso| inside the window, so edge values don't wash it out
    inside = in_window(pos, window)
    lim = np.nanmax(np.abs(sigma_iso[inside]))
    lim = max(lim, 1e-12) # the color scale needs a range even if the stress is zero everywhere
    fill = PolyCollection(cells, edgecolors=YELLOW, linewidths=1.0, cmap="PRGn", norm=TwoSlopeNorm(0.0, -lim, lim))
    fill.set_array(sigma_iso[which])
    right.add_collection(fill)
    fig.colorbar(fill, ax=right, shrink=0.8, label=r"isotropic stress $\sigma_{iso}$  (+ expanded, $-$ compressed)")
    right.set_title(r"Isotropic stress $\sigma_{iso}$ per cell")

    x_min, x_max, y_min, y_max = window
    for ax in (left, right):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    return fig
