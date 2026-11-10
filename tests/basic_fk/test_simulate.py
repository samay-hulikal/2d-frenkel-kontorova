import numpy as np

from src.basic_fk.simulate import run

A = np.sqrt(3)


def test_commensurate_s1_run_stays_locked():
    r = run(10, 10, A, A, 1.0, 0.3, fix_edges=True, start_spacing=A)
    s = r["summary"]
    assert s["converged"] and s["frac_left_window"] == 0.0 and np.isclose(s["E_after"], 0.0, atol=1e-12)


def test_free_patch_without_substrate_floats_out_of_wells():
    # s = 1.5, start in the minima, V0 = 0: the free patch shrinks to its own spacing, so particles away from the center leave their wells
    a_s = A * np.sqrt(1.5)
    r = run(14, 14, A, a_s, 1.0, 0.0, fix_edges=False, start_spacing=a_s, dt_snap=5.0)
    s = r["summary"]
    assert s["converged"] and s["n_flipped"] == 0
    assert s["frac_left_window"] > 0.3
    assert r["left"].shape == (s["n_snapshots"], 14 * 14) and not r["left"][0].any()
