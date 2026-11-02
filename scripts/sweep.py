#INFO: Parameter sweep over the substrate strength V0 and the area ratio s; one figure per run plus a summary table in outputs/sweep/


import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg") # save figures without opening windows
import matplotlib.pyplot as plt

from scripts.main import run
from src.plotting import plot_state

#INFO: Parameters; the bees' natural cell has side length 1, so their center-to-center spacing is a = sqrt(3)
NX, NY = 60, 60 # patch size; the central window (0.25x the area) holds about 30 x 30 bee cells
A = np.sqrt(3) # spring rest length (bees' natural spacing)
K = 1.0 # spring constant
V0_VALUES = np.round(np.arange(0.05, 2.05 + 1e-9, 0.25), 2) # 0.05, 0.30, ..., 2.05
S_VALUES = [0.5, 0.75, 1, 1.5, 2, 3] # substrate cell area / natural bee cell area
FIX_EDGES = False
START_AT_MINIMA = False # False = springs relaxed at the start (spacing a); True = every particle in a substrate minimum (spacing a_s)
F_TOL = 1e-6
WINDOW_FRACTION = 0.25

OUTPUT_DIR = Path("outputs") / "sweep"

COLUMNS = ["V0", "s", "a_s", "converged", "n_iter", "max_force", "E_before", "E_after", "E_springs_after", "E_substrate_after",
           "u_mean_window", "sigma_min", "sigma_mean", "sigma_max", "figure"]


def run_sweep(output_dir, start_at_minima):
    """
    Runs every (V0, s) pair, saves one figure per run and a summary table.

    Parameters
    ----------
    output_dir: Folder for the figures and summary.csv
    start_at_minima: If True, the patch starts at spacing a_s (every particle in a minimum); if False, at spacing a (springs relaxed)

    Returns
    ----------
    (none; prints one line per run and writes PNGs and summary.csv to output_dir)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    print(f"Sweep: {len(V0_VALUES)} V0 values x {len(S_VALUES)} s values = {len(V0_VALUES) * len(S_VALUES)} runs, A={A:.4f}, K={K}, "
          f"FIX_EDGES={FIX_EDGES}, start {'in the substrate minima (spacing a_s)' if start_at_minima else 'with springs relaxed (spacing a)'}")

    for V0 in V0_VALUES:
        for s in S_VALUES:
            a_s = A * np.sqrt(s) # cell areas scale as spacing^2, so a_s / a = sqrt(s)
            start_spacing = a_s if start_at_minima else A
            pos, sigma_iso, window, r = run(NX, NY, A, a_s, K, V0, FIX_EDGES, start_spacing, F_TOL, WINDOW_FRACTION)

            title = (f"{NX}×{NY} patch, central window · s = {s:g} (a_s/a = {a_s / A:.3g}), V0 = {V0:g}, k = {K:g}, "
                     f"edges {'fixed' if FIX_EDGES else 'free'}, start spacing {'a_s' if start_at_minima else 'a'}")
            fig = plot_state(pos, sigma_iso, window, a_s, title)
            name = f"V0_{V0:.2f}_s_{s:g}.png"
            fig.savefig(output_dir / name, dpi=110)
            plt.close(fig)

            row = {"V0": V0, "s": s, "a_s": a_s, "figure": name}
            row.update({c: r[c] for c in COLUMNS if c in r})
            rows.append(row)
            print(f"V0={V0:.2f} s={s:<4g} converged={r['converged']!s:5} iters={r['n_iter']:<5} max|F|={r['max_force']:.1e} "
                  f"E={r['E_after']:.2f} sigma in window [{r['sigma_min']:.3f}, {r['sigma_max']:.3f}] mean={r['sigma_mean']:.3f}")

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_bad = sum(not row["converged"] for row in rows)
    print(f"Done in {time.time() - start:.0f}s; {n_bad} run(s) not converged; figures and summary.csv in {output_dir}")


if __name__ == "__main__":
    run_sweep(OUTPUT_DIR, START_AT_MINIMA)
