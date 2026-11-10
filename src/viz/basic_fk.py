#INFO: Figures of the basic FK lattice on the substrate; drawing only, the physics lives in src/basic_fk/


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import Voronoi

from src.viz.substrate import substrate_cells
from src.window import in_window

BLUE = "#1f5fbf" # substrate cells
YELLOW = "#b8860b" # bees' (particle) Voronoi cells, dark yellow


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
