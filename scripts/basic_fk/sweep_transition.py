#INFO: Locked -> floating transition sweep: s evenly spaced, a few V0, free and fixed edges, every particle starting in a substrate minimum.
# Saves summary.csv and two plots (fraction out of starting well against s and against a / a_s) to outputs/basic_fk/transition/


import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg") # save figures without opening windows
import matplotlib.pyplot as plt

from src.basic_fk.simulate import run
from src.viz.transition import plot_transition

NX, NY = 60, 60
A = np.sqrt(3) # bees' natural spacing (cell side length 1)
K = 1.0
S_VALUES = np.linspace(0.6, 3.0, 13) # substrate cell area / bee cell area, evenly spaced (step 0.2)
V0_VALUES = [0.05, 0.25, 0.75] # PLACEHOLDER: to be chosen
EDGES = [False, True] # fix_edges
GAMMA, DT_SNAP, T_MAX, F_TOL = 1.0, 1.0, 2e4, 1e-6
WINDOW_FRACTION = 0.25
KICK_FRACTION = 1e-3 # kick size = KICK_FRACTION * a_s
SEED = 0

OUTPUT_DIR = Path("outputs") / "basic_fk" / "transition"

COLUMNS = ["V0", "s", "a_ratio", "fix_edges", "frac_left_window", "converged", "t_end", "max_force", "n_flipped", "n_window",
           "E_after", "sigma_mean"]


def run_sweep(output_dir, V0_values, s_values, edges):
    """
    Runs every (V0, s, edges) combination and saves the summary table and the two transition plots.

    Parameters
    ----------
    output_dir: Folder for summary.csv and the plots
    V0_values: Substrate peak heights to run
    s_values: Area ratios (substrate cell / natural bee cell) to run
    edges: List of fix_edges values to run

    Returns
    ----------
    rows: List of dictionaries, one per run
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    print(f"Sweep: {len(V0_values)} V0 x {len(s_values)} s x {len(edges)} edge types = {len(V0_values) * len(s_values) * len(edges)} runs")

    for fix_edges in edges:
        for V0 in V0_values:
            for s in s_values:
                a_s = A * np.sqrt(s) # cell areas scale as spacing^2
                t0 = time.time()
                r = run(NX, NY, A, a_s, K, V0, fix_edges, a_s, GAMMA, DT_SNAP, T_MAX, F_TOL, WINDOW_FRACTION,
                        KICK_FRACTION * a_s, SEED)["summary"]
                row = {"V0": V0, "s": s, "a_ratio": A / a_s, "fix_edges": fix_edges}
                row.update({c: r[c] for c in COLUMNS if c in r})
                rows.append(row)
                print(f"edges={'fixed' if fix_edges else 'free '} V0={V0:<6g} s={s:<5.2f} frac_left={r['frac_left_window']:.3f} "
                      f"converged={r['converged']!s:5} t_end={r['t_end']:<8.1f} flipped={r['n_flipped']} ({time.time() - t0:.0f}s)")

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    for x, name in (("s", "transition_vs_s.png"), ("a_ratio", "transition_vs_a_ratio.png")):
        fig = plot_transition(rows, x)
        fig.savefig(output_dir / name, dpi=120)
        plt.close(fig)

    n_bad = sum(not r["converged"] for r in rows)
    print(f"Done in {time.time() - start:.0f}s; {n_bad} run(s) not converged; summary.csv and plots in {output_dir}")
    return rows


if __name__ == "__main__":
    run_sweep(OUTPUT_DIR, V0_VALUES, S_VALUES, EDGES)
