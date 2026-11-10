#INFO: One basic FK run with overdamped dynamics; prints a short summary and saves the 2D figure to outputs/basic_fk/


from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg") # save figures without opening windows
import matplotlib.pyplot as plt

from src.basic_fk.simulate import run
from src.viz.basic_fk import plot_state
from src.viz.stack3d import plot_stack

#INFO: Parameters; the bees' natural cell has side length 1, so their center-to-center spacing is a = sqrt(3)
NX, NY = 60, 60 # patch size; the central window (0.25x the area) holds about 30 x 30 bee cells
A = np.sqrt(3) # spring rest length (bees' natural spacing)
S = 0.75 # substrate cell area / natural bee cell area
A_S = A * np.sqrt(S) # substrate spacing; cell areas scale as spacing^2, so a_s / a = sqrt(S)
K = 1.0 # spring constant
V0 = 0.1 # substrate peak height; placeholder until the values are chosen from the equations
FIX_EDGES = True # True pins the outer ring
START_SPACING = A_S # A_S = every particle starts in a substrate minimum (commensurate); A = springs relaxed
GAMMA = 1.0 # damping coefficient (sets the time unit)
DT_SNAP = 1.0 # time between snapshots
T_MAX = 1e4 # longest integration time
F_TOL = 1e-6 # stop once every mobile particle has |F_k| < F_TOL
WINDOW_FRACTION = 0.25 # fraction of the final patch's area in the central window
KICK_SIZE = 1e-3 * A_S # random starting displacement of each mobile particle; the commensurate start is an exact equilibrium without it
SEED = 0 # seed for the kick directions
N_LAYERS = 8 # layers in the 3D stack (snapshots evenly spaced in time)

OUTPUT_DIR = Path("outputs") / "basic_fk"


def main():
    """
    Runs one simulation, prints its summary and saves its figure.

    Parameters
    ----------
    (none; edit the constants at the top of the file)

    Returns
    ----------
    (none; prints a summary and writes a PNG and an interactive HTML to outputs/basic_fk/)
    """
    result = run(NX, NY, A, A_S, K, V0, FIX_EDGES, START_SPACING, GAMMA, DT_SNAP, T_MAX, F_TOL, WINDOW_FRACTION, KICK_SIZE, SEED)
    r = result["summary"]

    #INFO: Short summary
    print(f"Parameters: NX={NX} NY={NY} A={A:.4f} S={S} A_S={A_S:.4f} (A/A_S={A / A_S:.3f}) K={K} V0={V0}")
    print(f"            FIX_EDGES={FIX_EDGES} START_SPACING={START_SPACING:.4f} GAMMA={GAMMA} DT_SNAP={DT_SNAP} T_MAX={T_MAX:g} F_TOL={F_TOL}")
    print(f"            KICK_SIZE={KICK_SIZE:.2e} SEED={SEED}")
    print(f"Particles: {r['n_particles']} ({r['n_mobile']} mobile), bonds: {r['n_bonds']}, in window: {r['n_window']}")
    print(f"Dynamics: converged={r['converged']} t_end={r['t_end']:.2f} force calls={r['n_force_calls']} "
          f"max|F|={r['max_force']:.2e} snapshots={r['n_snapshots']} ({r['message']})")
    print(f"Flipped triangles (folding): {r['n_flipped']}; fraction out of starting well (window): {r['frac_left_window']:.3f}")
    print(f"Energy before: total={r['E_before']:.4f} springs={r['E_springs_before']:.4f} substrate={r['E_substrate_before']:.4f}")
    print(f"Energy after : total={r['E_after']:.4f} springs={r['E_springs_after']:.4f} substrate={r['E_substrate_after']:.4f}")
    print(f"Displacement |u|: max={r['u_max']:.4f} mean={r['u_mean']:.4f} (window mean={r['u_mean_window']:.4f})")
    print(f"Bond length: min={r['bond_min']:.4f} mean={r['bond_mean']:.4f} max={r['bond_max']:.4f} (A={A:.4f}, A_S={A_S:.4f})")
    print(f"sigma_iso in window: min={r['sigma_min']:.4f} mean={r['sigma_mean']:.4f} max={r['sigma_max']:.4f}")

    #INFO: Figure
    edges = "fixed" if FIX_EDGES else "free"
    title = (f"{NX}×{NY} patch, central window · s = {S:g} (a_s/a = {A_S / A:.3g}), V0 = {V0:g}, k = {K:g}, "
             f"edges {edges}, start spacing {START_SPACING / A:.3g} a")
    fig = plot_state(result["pos"], result["sigma_iso"], result["window"], A_S, title)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"state_s{S:g}_V0{V0:g}_{edges}_start{START_SPACING / A:.3g}a.png"
    fig.savefig(OUTPUT_DIR / name, dpi=110)
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_DIR / name}")

    #INFO: Interactive 3D stack (open the HTML in a browser)
    fig3d = plot_stack(result["times"], result["snapshots"], result["left"], result["window"], N_LAYERS, title)
    name3d = name.replace(".png", "_stack.html")
    fig3d.write_html(OUTPUT_DIR / name3d, include_plotlyjs="cdn")
    print(f"3D stack saved to {OUTPUT_DIR / name3d}")


if __name__ == "__main__":
    main()
