"""Finite triangular-lattice patches (particles = hexagonal-cell centres). No periodic boundaries."""

import numpy as np
from scipy.spatial import cKDTree


def triangular_patch(nx, ny, a=1.0):
    """Rectangular nx x ny patch of a triangular lattice with spacing a.

    Row j has y = j * a*sqrt(3)/2; odd rows are shifted by a/2 in x.
    Particle index = j * nx + i. Returns positions of shape (nx*ny, 2).
    """
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    # WARN: lattice geometry. Row spacing a*sqrt(3)/2 + half-shift on odd rows gives
    # equilateral triangles of side a; left/right edges are therefore zig-zag.
    x = a * (i + 0.5 * (j % 2))
    y = a * (np.sqrt(3) / 2) * j
    # WARN: end
    return np.column_stack([x.ravel(), y.ravel()]).astype(float)


def bonds_nn(pos, a=1.0, tol=1e-6):
    """Nearest-neighbour bonds: all pairs closer than a*(1+tol). Returns (M, 2) int array, i < j."""
    # WARN: bonds are found by distance, so call this on the *undeformed* lattice only.
    # The next-nearest distance is sqrt(3)*a, so tol must stay well below 0.73.
    pairs = cKDTree(pos).query_pairs(a * (1 + tol), output_type="ndarray")
    # WARN: end
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def edge_mask(nx, ny):
    """Boolean (nx*ny,) mask, True for the outer ring of the patch."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    return ((i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)).ravel()


def coordination(bonds, n):
    """Number of bonds per particle, shape (n,)."""
    return np.bincount(bonds.ravel(), minlength=n)
