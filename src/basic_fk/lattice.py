#INFO: Particles representing the centers of the hexagons connected by springs. The bees want the centers to be at certain distance because they want hexagons of a certain size.


import numpy as np
from scipy.spatial import cKDTree

def triangular_patch(nx, ny, a=1.0):
    """
    Rectangular nx x ny patch of a triangular lattice with spacing a, representing the centers of a hexagonal lattice.

    Parameters
    ----------
    nx: Number of centers along x
    ny: Number of centers along y
    a: Spacing between the centers of the hexagons

    Returns
    ----------
    pos: An array such that pos[j*nx + i] are the (x, y) coordinates of particle k = j*nx + i
    """
    # Index i goes along x (columns) and j goes along y (rows)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny)) # Creates an integer grid such that (i, j) gives the integer indices of the particles

    #INFO: Converting the integer indices to (x, y) positions
    x = a * (i + 0.5 * (j % 2))
    y = a * (np.sqrt(3) / 2) * j

    #INFO: pos[k] is a length-2 array [x_k, y_k], the position of particle k, where k = j*nx + i; pos[k, 0] is x_k and pos[k, 1] is y_k
    pos = np.column_stack([x.ravel(), y.ravel()]).astype(float) # ravel() flattens the grids
    return pos


def bonds_nn(pos, a=1.0, tol=1e-6):
    """
    Creates the nearest-neighbor bonds; run ONLY on an UNDEFORMED lattice.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k = i + j*nx
    a: Distance to check for neighbors
    tol: Tolerance within which neighbors must be considered

    Returns
    ----------
    bonds: bonds[l] outputs [p, q] indicating that particles p and q are connected by a bond
    """
    #INFO: Works only on an UNDEFORMED lattice
    pairs = cKDTree(pos).query_pairs(a * (1 + tol), output_type="ndarray") # asks the tree for all pairs (k, l) with |pos[k] − pos[l]| ≤ r, where r = a*(1+tol)

    bonds = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))] # bonds[l] returns [p, q] telling you that particles p and q are connected by bond number l
    return bonds

def edge_mask(nx, ny):
    """Identifying the edge vertices of the lattice.

    Parameters
    ----------
    nx: Number of particles along x
    ny: Number of particles along y

    Returns
    ----------
    mask: boolean array, mask[k] is True if particle k = j*nx + i is on the outer ring
    """
    i, j = np.meshgrid(np.arange(nx), np.arange(ny)) # Creates an integer grid such that (i, j) gives the integer indices of the particles

    mask = ((i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)).ravel() # mask[k] is true if particle k = i + j*nx is on the boundary and false otherwise
    return mask


def mobile_mask(nx, ny, fix_edges=True):
    """
    Identifying the particles that are allowed to move.

    Parameters
    ----------
    nx: Number of particles along x
    ny: Number of particles along y
    fix_edges: If True, the outer ring is pinned; if False, every particle moves

    Returns
    ----------
    mobile: Boolean array, mobile[k] is True if particle k = j*nx + i is allowed to move
    """
    if fix_edges:
        mobile = ~edge_mask(nx, ny) # Interior particles move, the outer ring is pinned
        return mobile

    mobile = np.ones(nx * ny, dtype=bool) # all particles move
    return mobile


def triangles(bonds):
    """
    The triangles of the spring network: every set of three particles that are all bonded to each other.

    Parameters
    ----------
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l

    Returns
    ----------
    tri: Array of shape (T, 3), tri[t] = [p, q, r] are the three particles of triangle t (p < q < r)
    """
    N = bonds.max() + 1
    neighbors = [set() for _ in range(N)] # neighbors[k] is the set of particles bonded to k
    for p, q in bonds:
        neighbors[p].add(q)
        neighbors[q].add(p)

    tri = []
    for p, q in bonds: # p < q (bonds_nn lists every bond that way)
        for r in neighbors[p] & neighbors[q]: # particles bonded to both p and q
            if r > q:
                tri.append([p, q, r]) # r > q > p, so each triangle is listed once
    tri = np.array(tri, dtype=int)
    return tri


def flipped_triangles(pos0, pos, tri):
    """
    Counts the triangles that are turned inside out compared with the starting patch (the network has folded over itself).

    Parameters
    ----------
    pos0: Array such that pos0[k] gives the STARTING (x, y) coordinates of particle k
    pos: Array such that pos[k] gives the current (x, y) coordinates of particle k
    tri: Array of shape (T, 3), the triangles from triangles(bonds)

    Returns
    ----------
    n_flipped: Number of triangles whose orientation has reversed
    """
    # WARN: orientation test. The signed area (r_q - r_p) x (r_r - r_p) is positive if p -> q -> r goes counterclockwise and
    # negative if clockwise; springs only fix lengths, so a triangle can pass through itself and come out mirrored, which flips the sign.
    def signed_area(P):
        ux = P[tri[:, 1], 0] - P[tri[:, 0], 0]
        uy = P[tri[:, 1], 1] - P[tri[:, 0], 1]
        vx = P[tri[:, 2], 0] - P[tri[:, 0], 0]
        vy = P[tri[:, 2], 1] - P[tri[:, 0], 1]
        return ux * vy - uy * vx
    n_flipped = int(np.sum(np.sign(signed_area(pos)) != np.sign(signed_area(pos0))))
    # WARN: end
    return n_flipped
