import numpy as np

from src.basic_fk.lattice import bonds_nn, edge_mask, flipped_triangles, mobile_mask, triangles, triangular_patch


def coordination(bonds, n):
    """
    Number of bonds per particle (coordination number z).

    Parameters
    ----------
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    n: Number of particles

    Returns
    ----------
    z: Array of shape (n,) such that z[k] is the number of bonds attached to particle k
    """
    return np.bincount(bonds.ravel(), minlength=n)


def test_patch_size_and_spacing():
    pos = triangular_patch(5, 4, a=2.0)
    assert pos.shape == (20, 2)
    bonds = bonds_nn(pos, a=2.0)
    lengths = np.linalg.norm(pos[bonds[:, 1]] - pos[bonds[:, 0]], axis=1)
    assert np.allclose(lengths, 2.0)


def test_interior_has_six_bonds_edges_fewer():
    nx, ny = 6, 5
    pos = triangular_patch(nx, ny)
    z = coordination(bonds_nn(pos), nx * ny)
    edge = edge_mask(nx, ny)
    assert np.all(z[~edge] == 6)
    assert np.all(z[edge] < 6)


def test_edge_mask_is_outer_ring():
    nx, ny = 6, 5
    edge = edge_mask(nx, ny).reshape(ny, nx)
    assert edge[0].all() and edge[-1].all() and edge[:, 0].all() and edge[:, -1].all()
    assert not edge[1:-1, 1:-1].any()
    assert edge.sum() == 2 * nx + 2 * (ny - 2)


def test_mobile_mask_fixed_and_free_edges():
    nx, ny = 6, 5
    assert np.array_equal(mobile_mask(nx, ny, fix_edges=True), ~edge_mask(nx, ny))
    free = mobile_mask(nx, ny, fix_edges=False)
    assert free.shape == (nx * ny,) and free.all()


def test_triangle_count_and_no_flips_when_undeformed():
    nx, ny = 7, 6
    pos = triangular_patch(nx, ny, a=1.3)
    tri = triangles(bonds_nn(pos, a=1.3))
    assert tri.shape == (2 * (nx - 1) * (ny - 1), 3) # each strip between two rows holds 2(nx - 1) triangles
    assert flipped_triangles(pos, 1.7 * pos, tri) == 0 # uniform stretch keeps every orientation


def test_mirroring_a_particle_flips_its_triangles():
    nx, ny = 7, 6
    pos0 = triangular_patch(nx, ny, a=1.0)
    tri = triangles(bonds_nn(pos0, a=1.0))
    pos = pos0.copy()
    k = 2 * nx + 3 # interior particle, in 6 triangles
    pos[k, 1] -= np.sqrt(3) # push it down past its lower neighbors (a mirror image through that row)
    assert flipped_triangles(pos0, pos, tri) > 0
    assert flipped_triangles(pos0, pos, tri) <= 6
