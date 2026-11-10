import numpy as np

from src.basic_fk.energy import total_energy, total_force
from src.basic_fk.lattice import bonds_nn, triangular_patch

K, A, V0, A_S = 2.0, 1.3, 0.7, 1.2 # non-unit values so wrong scalings show up


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
