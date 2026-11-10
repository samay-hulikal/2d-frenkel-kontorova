import numpy as np

from src.basic_fk.lattice import bonds_nn, triangular_patch
from src.basic_fk.springs import spring_energy, spring_force

K, A = 2.0, 1.3 # non-unit values so wrong scalings show up


def test_undeformed_lattice_has_no_energy_or_force():
    pos = triangular_patch(6, 5, a=A)
    bonds = bonds_nn(pos, a=A)
    assert np.isclose(spring_energy(pos, bonds, K, A), 0.0)
    assert np.allclose(spring_force(pos, bonds, K, A), 0.0)


def test_single_stretched_bond():
    pos = np.array([[0.0, 0.0], [A + 0.5, 0.0]]) # bond stretched by 0.5 along x
    bonds = np.array([[0, 1]])
    assert np.isclose(spring_energy(pos, bonds, K, A), (K / 2) * 0.5**2)
    F = spring_force(pos, bonds, K, A)
    assert np.allclose(F[1], [-K * 0.5, 0.0]) # particle 1 pulled back towards particle 0
    assert np.allclose(F[0], [K * 0.5, 0.0]) # particle 0 pulled towards particle 1


def test_forces_add_up_to_zero():
    rng = np.random.default_rng(0)
    pos0 = triangular_patch(6, 5, a=A)
    bonds = bonds_nn(pos0, a=A)
    pos = pos0 + rng.normal(scale=0.1, size=pos0.shape)
    assert np.allclose(spring_force(pos, bonds, K, A).sum(axis=0), 0.0)


def test_force_is_minus_numerical_gradient():
    rng = np.random.default_rng(1)
    pos0 = triangular_patch(5, 4, a=A)
    bonds = bonds_nn(pos0, a=A)
    pos = pos0 + rng.normal(scale=0.1, size=pos0.shape)
    h = 1e-6
    num = np.zeros_like(pos)
    for n in range(pos.shape[0]):
        for d in range(2):
            plus, minus = pos.copy(), pos.copy()
            plus[n, d] += h
            minus[n, d] -= h
            num[n, d] = -(spring_energy(plus, bonds, K, A) - spring_energy(minus, bonds, K, A)) / (2 * h)
    assert np.allclose(spring_force(pos, bonds, K, A), num, atol=1e-6)
