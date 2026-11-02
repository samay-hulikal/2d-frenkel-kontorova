import numpy as np

from src.lattice import bonds_nn, mobile_mask, triangular_patch
from src.relax import relax, total_energy, total_force

K, A, V0, A_S = 2.0, 1.3, 0.7, 1.2 # non-unit values so wrong scalings show up
NX, NY = 8, 7


def test_total_force_is_minus_numerical_gradient():
    rng = np.random.default_rng(0)
    pos0 = triangular_patch(5, 4, a=A_S)
    bonds = bonds_nn(pos0, a=A_S)
    pos = pos0 + rng.normal(scale=0.1, size=pos0.shape)
    h = 1e-6
    num = np.zeros_like(pos)
    for n in range(pos.shape[0]):
        for d in range(2):
            plus, minus = pos.copy(), pos.copy()
            plus[n, d] += h
            minus[n, d] -= h
            num[n, d] = -(total_energy(plus, bonds, K, A, V0, A_S) - total_energy(minus, bonds, K, A, V0, A_S)) / (2 * h)
    assert np.allclose(total_force(pos, bonds, K, A, V0, A_S), num, atol=1e-6)


def test_commensurate_start_does_not_move():
    pos0 = triangular_patch(NX, NY, a=A_S)
    bonds = bonds_nn(pos0, a=A_S)
    pos, info = relax(pos0, bonds, mobile_mask(NX, NY, fix_edges=False), K, A_S, V0, A_S) # a = a_s
    assert info["converged"]
    assert np.allclose(pos, pos0, atol=1e-8)
    assert np.isclose(info["energy"], 0.0, atol=1e-12)


def test_perturbed_start_relaxes_and_lowers_energy():
    rng = np.random.default_rng(1)
    pos0 = triangular_patch(NX, NY, a=A_S)
    bonds = bonds_nn(pos0, a=A_S)
    start = pos0 + rng.normal(scale=0.15, size=pos0.shape)
    for fix_edges in (True, False):
        mobile = mobile_mask(NX, NY, fix_edges)
        pos, info = relax(start, bonds, mobile, K, A, V0, A_S)
        assert info["converged"], info["message"]
        assert info["max_force"] < 1e-6
        assert info["energy"] < total_energy(start, bonds, K, A, V0, A_S)


def test_pinned_particles_stay_put():
    rng = np.random.default_rng(2)
    pos0 = triangular_patch(NX, NY, a=A_S)
    bonds = bonds_nn(pos0, a=A_S)
    start = pos0 + rng.normal(scale=0.15, size=pos0.shape)
    mobile = mobile_mask(NX, NY, fix_edges=True)
    pos, info = relax(start, bonds, mobile, K, A, V0, A_S)
    assert np.array_equal(pos[~mobile], start[~mobile])
