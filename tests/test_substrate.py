import numpy as np

from src.basic_fk.lattice import triangular_patch
from src.substrate import reciprocal_vectors, substrate_energy, substrate_force

V0, A_S = 2.0, 1.3 # non-unit values so wrong scalings show up


def test_reciprocal_vectors_length_and_sum():
    G = reciprocal_vectors(A_S)
    assert np.allclose(np.linalg.norm(G, axis=1), 2 / (np.sqrt(3) * A_S))
    assert np.allclose(G.sum(axis=0), 0.0)


def test_commensurate_patch_sits_in_minima():
    pos = triangular_patch(6, 5, a=A_S)
    assert np.allclose(substrate_energy(pos, V0, A_S), 0.0, atol=1e-12)
    assert np.allclose(substrate_force(pos, V0, A_S), 0.0, atol=1e-10)


def test_maximum_and_saddle_values():
    top = np.array([[A_S / 2, A_S / (2 * np.sqrt(3))]]) # triangle center
    saddle = np.array([[A_S / 2, 0.0]]) # midway between two neighboring minima
    assert np.allclose(substrate_energy(top, V0, A_S), V0)
    assert np.allclose(substrate_energy(saddle, V0, A_S), 8 * V0 / 9)
    assert np.allclose(substrate_force(np.vstack([top, saddle]), V0, A_S), 0.0, atol=1e-10)


def test_force_is_minus_numerical_gradient():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-3, 3, size=(20, 2))
    h = 1e-6
    num = np.zeros_like(pos)
    for d in range(2):
        step = np.zeros(2)
        step[d] = h
        num[:, d] = -(substrate_energy(pos + step, V0, A_S) - substrate_energy(pos - step, V0, A_S)) / (2 * h)
    assert np.allclose(substrate_force(pos, V0, A_S), num, atol=1e-6)


def test_sixfold_symmetry():
    rng = np.random.default_rng(1)
    pos = rng.uniform(-3, 3, size=(20, 2))
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    rot = np.array([[c, -s], [s, c]])
    assert np.allclose(substrate_energy(pos @ rot.T, V0, A_S), substrate_energy(pos, V0, A_S))
