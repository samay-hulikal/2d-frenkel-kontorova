#INFO: Local stress in the bees' lattice from the springs; each particle's stress is averaged over its own Voronoi cell


import numpy as np
from scipy.spatial import Voronoi

from src.springs import bond_vectors


def voronoi_areas(pos):
    """
    Area of each particle's Voronoi cell.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k

    Returns
    ----------
    areas: Array of shape (N,), areas[k] is the area of the Voronoi cell of particle k; NaN if the cell is unbounded
    """
    vor = Voronoi(pos)
    N = pos.shape[0]
    areas = np.full(N, np.nan)

    for k in range(N):
        region = vor.regions[vor.point_region[k]] # indices of the cell's corners, in order around the cell
        if len(region) == 0 or -1 in region:
            continue # -1 marks a corner at infinity, i.e. an unbounded cell on the outside of the patch

        #INFO: Shoelace formula for the area of a polygon with corners (x_i, y_i) listed in order
        x = vor.vertices[region, 0]
        y = vor.vertices[region, 1]
        areas[k] = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return areas


def isotropic_stress(pos, bonds, areas, k=1.0, a=1.0):
    """
    Isotropic spring stress of each particle, sigma_iso[k] = (1/2) tr sigma_k; positive = expanded, negative = compressed.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    areas: Array such that areas[k] is the Voronoi cell area of particle k
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)

    Returns
    ----------
    sigma_iso: Array of shape (N,), sigma_iso[k] is the isotropic stress of particle k; NaN where areas[k] is NaN
    """
    dx, dy, length = bond_vectors(pos, bonds)
    dl = length - a

    # WARN: virial stress. sigma = -(1/A) sum_bonds d (x) f, with d = r_q - r_p and f = -k dl d/|d| the force on q.
    # This gives sigma = (1/A) sum_bonds k dl |d| (d/|d|) (x) (d/|d|), so a stretched bond (dl > 0) gives positive stress,
    # and tr sigma = (1/A) sum_bonds k dl |d|. Only the springs count: the substrate is an outside force, not a stress.
    trace_term = k * dl * length # contribution of bond l to A * tr(sigma), shape (M,)
    # WARN: end

    # WARN: splitting bonds between particles. Each bond is shared by its two particles, so half of it goes to each;
    # then sigma_iso[k] = (1/2) tr sigma_k = (1 / (2 A_k)) * (sum over bonds of k of trace_term / 2).
    N = pos.shape[0]
    T = np.zeros(N) # T[k] = A_k * tr(sigma_k)
    np.add.at(T, bonds[:, 0], 0.5 * trace_term) # half to p
    np.add.at(T, bonds[:, 1], 0.5 * trace_term) # half to q
    sigma_iso = T / (2 * areas)
    # WARN: end
    return sigma_iso
