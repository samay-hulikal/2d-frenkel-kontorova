#INFO: Which substrate well each particle sits in; tracks the commensurate (locked) to incommensurate (floating) transition


import numpy as np

from src.substrate import reciprocal_vectors


def nearest_minimum(pos, a_s=1.0):
    """
    Integer label (i, j) of the substrate minimum closest to each point; the minimum sits at i v_1 + j v_2.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of point k
    a_s: Spacing between neighboring substrate minima

    Returns
    ----------
    ij: Integer array of shape (N, 2), ij[k] = (i, j) labels the minimum closest to point k
    """
    v1 = a_s * np.array([1.0, 0.0])
    v2 = a_s * np.array([0.5, np.sqrt(3) / 2])
    G = reciprocal_vectors(a_s)

    # WARN: nearest minimum. The well of a minimum is its hexagonal substrate cell, i.e. the set of points closer to it than to any
    # other minimum. Since v_i . G_j = delta_ij, the lattice coordinates of r are (G_1 . r, G_2 . r); the closest minimum is one of
    # the 4 corners of the rhombus (floor/ceil of each coordinate) that contains r, so check those 4 and keep the nearest.
    c1 = pos @ G[0]
    c2 = pos @ G[1]
    i0 = np.floor(c1)
    j0 = np.floor(c2)
    best_d2 = np.full(pos.shape[0], np.inf)
    ij = np.zeros((pos.shape[0], 2), dtype=int)
    for di in (0, 1):
        for dj in (0, 1):
            i = i0 + di
            j = j0 + dj
            mx = i * v1[0] + j * v2[0] # position of candidate minimum
            my = i * v1[1] + j * v2[1]
            d2 = (pos[:, 0] - mx)**2 + (pos[:, 1] - my)**2
            closer = d2 < best_d2
            best_d2[closer] = d2[closer]
            ij[closer, 0] = i[closer]
            ij[closer, 1] = j[closer]
    # WARN: end
    return ij


def fraction_left_well(pos0, pos, a_s=1.0, which=None):
    """
    Fraction of particles that are no longer in the substrate well they started in.

    Parameters
    ----------
    pos0: Array such that pos0[k] gives the STARTING (x, y) coordinates of particle k
    pos: Array such that pos[k] gives the current (x, y) coordinates of particle k
    a_s: Spacing between neighboring substrate minima
    which: Optional boolean array, which[k] is True if particle k should be counted (e.g. inside the central window); default all

    Returns
    ----------
    fraction: Number of counted particles that left their starting well divided by the number of counted particles
    left: Boolean array, left[k] is True if particle k is in a different well than at the start
    """
    left = np.any(nearest_minimum(pos, a_s) != nearest_minimum(pos0, a_s), axis=1)
    if which is None:
        which = np.ones(pos.shape[0], dtype=bool)
    fraction = left[which].mean()
    return fraction, left
