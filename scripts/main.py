#INFO: Build the patch, relax it on the substrate, print a short summary and save the figure to outputs/


from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg") # save figures without opening windows
import matplotlib.pyplot as plt

from src.lattice import bonds_nn, mobile_mask, triangular_patch
from src.plotting import central_window, in_window, plot_state
from src.relax import relax
from src.springs import bond_vectors, spring_energy
from src.stress import isotropic_stress, voronoi_areas
from src.substrate import substrate_energy

#INFO: Parameters; the bees' natural cell has side length 1, so their center-to-center spacing is a = sqrt(3)
NX, NY = 60, 60 # patch size; the central window (0.25x the area) holds about 30 x 30 bee cells
A = np.sqrt(3) # spring rest length (bees' natural spacing)
S = 0.75 # substrate cell area / natural bee cell area
A_S = A * np.sqrt(S) # substrate spacing; cell areas scale as spacing^2, so a_s / a = sqrt(S)
K = 1.0 # spring constant
V0 = 0.1 # substrate peak height; placeholder until the values are chosen from the equations
FIX_EDGES = False # True pins the outer ring
START_SPACING = A # spacing of the starting patch; A = springs relaxed, A_S = every particle in a substrate minimum
F_TOL = 1e-6 # stop once every mobile particle has |F_k| < F_TOL
WINDOW_FRACTION = 0.25 # fraction of the patch's area shown in the figure

OUTPUT_DIR = Path("outputs")


def run(nx, ny, a, a_s, k, V0, fix_edges, start_spacing, f_tol=1e-6, window_fraction=0.25):
    """
    Builds the patch, relaxes it and computes the stress and a summary.

    Parameters
    ----------
    nx: Number of particles along x
    ny: Number of particles along y
    a: Rest length of the springs (natural cell spacing)
    a_s: Spacing between neighboring substrate minima
    k: Spring constant
    V0: Peak height of the substrate potential
    fix_edges: If True, the outer ring is pinned
    start_spacing: Spacing of the starting patch
    f_tol: Stop once every mobile particle has |F_k| < f_tol
    window_fraction: Fraction of the patch's area inside the central window

    Returns
    ----------
    pos: Array of shape (N, 2), the relaxed positions
    sigma_iso: Array of shape (N,), the isotropic stress of each particle
    window: Tuple (x_min, x_max, y_min, y_max), the central window
    summary: Dictionary of numbers describing the run (energies, convergence, displacement, bond lengths, stress in window)
    """
    pos0 = triangular_patch(nx, ny, start_spacing)
    bonds = bonds_nn(pos0, start_spacing) # bonds from the undeformed starting patch, never rewired
    mobile = mobile_mask(nx, ny, fix_edges)

    pos, info = relax(pos0, bonds, mobile, k, a, V0, a_s, f_tol)

    sigma_iso = isotropic_stress(pos, bonds, voronoi_areas(pos), k, a)
    window = central_window(pos0, window_fraction)
    inside = in_window(pos, window)

    u = np.sqrt(np.sum((pos - pos0)**2, axis=1)) # how far each particle moved
    _, _, length = bond_vectors(pos, bonds)
    E0_int, E0_sub = spring_energy(pos0, bonds, k, a), np.sum(substrate_energy(pos0, V0, a_s))
    E_int, E_sub = spring_energy(pos, bonds, k, a), np.sum(substrate_energy(pos, V0, a_s))

    summary = {
        "n_particles": pos.shape[0], "n_mobile": int(mobile.sum()), "n_bonds": bonds.shape[0], "n_window": int(inside.sum()),
        "converged": info["converged"], "n_iter": info["n_iter"], "max_force": info["max_force"], "message": info["message"],
        "E_before": E0_int + E0_sub, "E_springs_before": E0_int, "E_substrate_before": E0_sub,
        "E_after": E_int + E_sub, "E_springs_after": E_int, "E_substrate_after": E_sub,
        "u_max": u.max(), "u_mean": u.mean(), "u_mean_window": u[inside].mean(),
        "bond_min": length.min(), "bond_mean": length.mean(), "bond_max": length.max(),
        "sigma_min": np.nanmin(sigma_iso[inside]), "sigma_mean": np.nanmean(sigma_iso[inside]), "sigma_max": np.nanmax(sigma_iso[inside]),
    }
    return pos, sigma_iso, window, summary


def main():
    """
    Runs one relaxation and saves its figure.

    Parameters
    ----------
    (none; edit the constants at the top of the file)

    Returns
    ----------
    (none; prints a summary and writes a PNG to outputs/)
    """
    pos, sigma_iso, window, r = run(NX, NY, A, A_S, K, V0, FIX_EDGES, START_SPACING, F_TOL, WINDOW_FRACTION)

    #INFO: Short summary
    print(f"Parameters: NX={NX} NY={NY} A={A:.4f} S={S} A_S={A_S:.4f} (A/A_S={A / A_S:.3f}) K={K} V0={V0} (V0/(K A^2)={V0 / (K * A**2):.3f})")
    print(f"            FIX_EDGES={FIX_EDGES} START_SPACING={START_SPACING:.4f} F_TOL={F_TOL}")
    print(f"Particles: {r['n_particles']} ({r['n_mobile']} mobile), bonds: {r['n_bonds']}, in window: {r['n_window']}")
    print(f"Relax: converged={r['converged']} iterations={r['n_iter']} max|F|={r['max_force']:.2e} ({r['message']})")
    print(f"Energy before: total={r['E_before']:.4f} springs={r['E_springs_before']:.4f} substrate={r['E_substrate_before']:.4f}")
    print(f"Energy after : total={r['E_after']:.4f} springs={r['E_springs_after']:.4f} substrate={r['E_substrate_after']:.4f}")
    print(f"Displacement |u|: max={r['u_max']:.4f} mean={r['u_mean']:.4f} (window mean={r['u_mean_window']:.4f})")
    print(f"Bond length: min={r['bond_min']:.4f} mean={r['bond_mean']:.4f} max={r['bond_max']:.4f} (A={A:.4f}, A_S={A_S:.4f})")
    print(f"sigma_iso in window: min={r['sigma_min']:.4f} mean={r['sigma_mean']:.4f} max={r['sigma_max']:.4f}")

    #INFO: Figure
    title = (f"{NX}×{NY} patch, central window · s = {S:g} (a_s/a = {A_S / A:.3g}), V0 = {V0:g}, k = {K:g}, "
             f"edges {'fixed' if FIX_EDGES else 'free'}, start spacing {START_SPACING / A:.3g} a")
    fig = plot_state(pos, sigma_iso, window, A_S, title)
    OUTPUT_DIR.mkdir(exist_ok=True)
    name = f"state_s{S:g}_V0{V0:g}_{'fixed' if FIX_EDGES else 'free'}_start{START_SPACING / A:.3g}a.png"
    fig.savefig(OUTPUT_DIR / name, dpi=110)
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
