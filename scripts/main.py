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

#INFO: Parameters; k = a_s = 1 fix the units of energy and length, so the physics is set by A / A_S and V0 / (K A_S^2)
NX, NY = 60, 60 # patch size; the central window (0.25x the area) holds about 30 x 30 cells
A_S = 1.0 # substrate spacing
A = 1.1 # spring rest length (bees' natural spacing)
K = 1.0 # spring constant
V0 = 0.1 # substrate peak height; placeholder until the values are chosen from the equations
FIX_EDGES = False # True pins the outer ring
START_SPACING = A_S # spacing of the starting patch; A_S = every particle in a substrate minimum
F_TOL = 1e-6 # stop once every mobile particle has |F_k| < F_TOL
WINDOW_FRACTION = 0.25 # fraction of the patch's area shown in the figure

OUTPUT_DIR = Path("outputs")


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
    pos0 = triangular_patch(NX, NY, START_SPACING)
    bonds = bonds_nn(pos0, START_SPACING) # bonds from the undeformed starting patch, never rewired
    mobile = mobile_mask(NX, NY, FIX_EDGES)

    pos, info = relax(pos0, bonds, mobile, K, A, V0, A_S, F_TOL)

    sigma_iso = isotropic_stress(pos, bonds, voronoi_areas(pos), K, A)
    window = central_window(pos0, WINDOW_FRACTION)
    inside = in_window(pos, window)

    #INFO: Short summary
    u = np.sqrt(np.sum((pos - pos0)**2, axis=1)) # how far each particle moved
    _, _, length = bond_vectors(pos, bonds)
    print(f"Parameters: NX={NX} NY={NY} A_S={A_S} A={A} (A/A_S={A / A_S:.3f}) K={K} V0={V0} (V0/(K A_S^2)={V0 / (K * A_S**2):.3f})")
    print(f"            FIX_EDGES={FIX_EDGES} START_SPACING={START_SPACING} F_TOL={F_TOL}")
    print(f"Particles: {pos.shape[0]} ({mobile.sum()} mobile), bonds: {bonds.shape[0]}, in window: {inside.sum()}")
    print(f"Relax: converged={info['converged']} iterations={info['n_iter']} max|F|={info['max_force']:.2e} ({info['message']})")
    for name, p in (("before", pos0), ("after", pos)):
        E_int = spring_energy(p, bonds, K, A)
        E_sub = np.sum(substrate_energy(p, V0, A_S))
        print(f"Energy {name:6}: total={E_int + E_sub:.4f} springs={E_int:.4f} substrate={E_sub:.4f}")
    print(f"Displacement |u|: max={u.max():.4f} mean={u.mean():.4f} (window mean={u[inside].mean():.4f})")
    print(f"Bond length: min={length.min():.4f} mean={length.mean():.4f} max={length.max():.4f} (A={A}, A_S={A_S})")
    print(f"sigma_iso in window: min={np.nanmin(sigma_iso[inside]):.4f} mean={np.nanmean(sigma_iso[inside]):.4f} max={np.nanmax(sigma_iso[inside]):.4f}")

    #INFO: Figure
    title = (f"{NX}×{NY} patch, central window · a/a_s = {A / A_S:.3g}, V0/(k a_s²) = {V0 / (K * A_S**2):.3g}, "
             f"edges {'fixed' if FIX_EDGES else 'free'}, start spacing {START_SPACING:.3g}")
    fig = plot_state(pos, sigma_iso, window, A_S, title)
    OUTPUT_DIR.mkdir(exist_ok=True)
    name = f"state_a{A:g}_V0{V0:g}_{'fixed' if FIX_EDGES else 'free'}_start{START_SPACING:g}.png"
    fig.savefig(OUTPUT_DIR / name, dpi=110)
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
