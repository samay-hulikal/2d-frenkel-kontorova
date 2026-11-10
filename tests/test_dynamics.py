import numpy as np

from src.basic_fk.energy import total_energy, total_force
from src.basic_fk.lattice import bonds_nn, flipped_triangles, mobile_mask, triangles, triangular_patch
from src.dynamics import overdamped

K, A, V0, A_S = 2.0, 1.3, 0.7, 1.2 # non-unit values so wrong scalings show up
NX, NY = 8, 7


def test_harmonic_decay_matches_exact_solution():
    # F = -c (r - r_eq) gives r(t) = r_eq + (r0 - r_eq) exp(-c t / gamma); particle 1 is pinned
    c, gamma = 1.7, 0.6
    r_eq = np.array([[0.3, -0.2], [5.0, 5.0]])
    pos0 = np.array([[2.0, 1.0], [4.0, 4.0]])
    mobile = np.array([True, False])
    times, snaps, info = overdamped(lambda P: -c * (P - r_eq), pos0, mobile, gamma, dt_snap=0.25, t_max=2.0, f_tol=1e-12, rtol=1e-9, atol=1e-12)
    exact = r_eq[0] + (pos0[0] - r_eq[0]) * np.exp(-c * times / gamma)[:, None]
    assert np.allclose(snaps[:, 0], exact, atol=1e-7)
    assert np.allclose(snaps[:, 1], pos0[1]) # pinned particle never moves
    assert np.allclose(times, np.arange(0, 2.0001, 0.25)) and not info["converged"]


def test_stops_once_force_below_tolerance():
    r_eq = np.zeros((1, 2))
    times, snaps, info = overdamped(lambda P: -(P - r_eq), np.array([[1.0, 0.0]]), np.array([True]), dt_snap=1.0, t_max=100.0, f_tol=1e-3)
    assert info["converged"] and info["max_force"] < 1.01e-3
    assert np.isclose(info["t_end"], np.log(1e3), rtol=1e-3) # |F| = exp(-t) reaches 1e-3 at t = ln(1000)
    assert times[-1] == info["t_end"]


def test_commensurate_start_does_not_move():
    pos0 = triangular_patch(NX, NY, a=A_S)
    bonds = bonds_nn(pos0, a=A_S)
    force = lambda P: total_force(P, bonds, K, A_S, V0, A_S) # a = a_s: every particle in a minimum with relaxed springs
    times, snaps, info = overdamped(force, pos0, mobile_mask(NX, NY, False))
    assert info["converged"] and len(times) == 1
    assert np.array_equal(snaps[-1], pos0)


def test_energy_never_increases_and_minimum_is_reached():
    rng = np.random.default_rng(1)
    pos0 = triangular_patch(NX, NY, a=A_S)
    bonds = bonds_nn(pos0, a=A_S)
    start = pos0 + rng.normal(scale=0.15, size=pos0.shape)
    for fix_edges in (True, False):
        mobile = mobile_mask(NX, NY, fix_edges)
        times, snaps, info = overdamped(lambda P: total_force(P, bonds, K, A, V0, A_S), start, mobile, dt_snap=0.5)
        E = np.array([total_energy(P, bonds, K, A, V0, A_S) for P in snaps])
        assert info["converged"], info["message"]
        assert np.all(np.diff(E) <= 1e-9)
        assert np.array_equal(snaps[-1][~mobile], start[~mobile])


def test_stretched_free_patch_shrinks_to_its_own_spacing_without_folding():
    # no substrate, 30% stretched start: the free patch should relax to E = 0 and keep every triangle's orientation
    n = 12
    pos0 = triangular_patch(n, n, a=1.3 * A)
    bonds = bonds_nn(pos0, a=1.3 * A)
    times, snaps, info = overdamped(lambda P: total_force(P, bonds, K, A, 0.0, A_S), pos0, mobile_mask(n, n, False), dt_snap=5.0)
    assert info["converged"]
    assert np.isclose(total_energy(snaps[-1], bonds, K, A, 0.0, A_S), 0.0, atol=1e-9)
    assert flipped_triangles(pos0, snaps[-1], triangles(bonds)) == 0
