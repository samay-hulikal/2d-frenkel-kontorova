import numpy as np

from src.lattice import bonds_nn, coordination, edge_mask, triangular_patch


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
