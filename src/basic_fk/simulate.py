#INFO: One basic FK run: build the patch, integrate the overdamped dynamics, compute stress, wells and a summary


import numpy as np

from src.basic_fk.energy import total_force
from src.basic_fk.lattice import bonds_nn, flipped_triangles, mobile_mask, triangles, triangular_patch
from src.basic_fk.springs import bond_vectors, spring_energy
from src.basic_fk.stress import isotropic_stress, voronoi_areas
from src.basic_fk.wells import fraction_left_well
from src.dynamics import overdamped
from src.substrate import substrate_energy
from src.window import central_window, in_window


def kick(pos, mobile, size, seed=0):
    """
    Moves every mobile particle by the same small distance in its own random direction (breaks the symmetry of an exact equilibrium).

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    mobile: Boolean array, mobile[k] is True if particle k is allowed to move
    size: Length of each displacement
    seed: Seed of the random number generator, so a run can be repeated exactly

    Returns
    ----------
    kicked: Array of shape (N, 2), the displaced positions (pinned particles unchanged)
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, size=pos.shape[0]) # uniform direction for each particle
    kicked = pos.copy()
    kicked[mobile, 0] += size * np.cos(angle[mobile])
    kicked[mobile, 1] += size * np.sin(angle[mobile])
    return kicked


def run(nx, ny, a, a_s, k, V0, fix_edges, start_spacing, gamma=1.0, dt_snap=1.0, t_max=1e4, f_tol=1e-6, window_fraction=0.25,
        kick_size=0.0, seed=0):
    """
    Builds the patch, runs the overdamped dynamics to a local minimum (or t_max) and summarizes the result.

    Parameters
    ----------
    nx: Number of particles along x
    ny: Number of particles along y
    a: Rest length of the springs (natural cell spacing)
    a_s: Spacing between neighboring substrate minima
    k: Spring constant
    V0: Peak height of the substrate potential
    fix_edges: If True, the outer ring is pinned
    start_spacing: Spacing of the starting patch (a_s = every particle in a substrate minimum)
    gamma: Damping coefficient
    dt_snap: Time between saved snapshots
    t_max: Longest time to integrate
    f_tol: Stop once every mobile particle has |F_k| < f_tol
    window_fraction: Fraction of the final patch's area inside the central window
    kick_size: Length of the random starting displacement of each mobile particle (0 = no kick)
    seed: Seed for the kick directions

    Returns
    ----------
    result: Dictionary with pos0 (unkicked lattice; wells and displacements are measured from it), times, snapshots (S, N, 2), pos (final), left (S, N) (True = particle outside its starting well),
            sigma_iso (final, per particle), window, inside (final positions in the window) and summary (dictionary of numbers)
    """
    pos0 = triangular_patch(nx, ny, start_spacing)
    bonds = bonds_nn(pos0, start_spacing) # bonds from the undeformed starting patch, never rewired
    mobile = mobile_mask(nx, ny, fix_edges)

    # WARN: the starting kick. The commensurate start (every particle in a minimum, uniform spacing) has zero force on every particle
    # for any s, so without a kick the dynamics never leave it, even if it is unstable. A kick of size << a_s tests that stability:
    # it decays if the start is a stable minimum and grows if not. Wells and displacements are measured from the UNKICKED lattice.
    start = kick(pos0, mobile, kick_size, seed)
    # WARN: end

    force = lambda P: total_force(P, bonds, k, a, V0, a_s)
    times, snapshots, info = overdamped(force, start, mobile, gamma, dt_snap, t_max, f_tol)
    pos = snapshots[-1]

    window = central_window(pos, window_fraction) # from the FINAL patch, so it holds about the same particles whatever the patch did
    inside = in_window(pos, window)
    left = np.array([fraction_left_well(pos0, P, a_s)[1] for P in snapshots]) # left[n, k]: particle k out of its well at times[n]
    frac_left = left[-1][inside].mean()

    sigma_iso = isotropic_stress(pos, bonds, voronoi_areas(pos), k, a)
    n_flipped = flipped_triangles(pos0, pos, triangles(bonds)) # should be 0; > 0 means the network folded over itself
    u = np.sqrt(np.sum((pos - pos0)**2, axis=1)) # how far each particle moved
    _, _, length = bond_vectors(pos, bonds)
    E0_int, E0_sub = spring_energy(start, bonds, k, a), np.sum(substrate_energy(start, V0, a_s))
    E_int, E_sub = spring_energy(pos, bonds, k, a), np.sum(substrate_energy(pos, V0, a_s))

    summary = {
        "n_particles": pos.shape[0], "n_mobile": int(mobile.sum()), "n_bonds": bonds.shape[0], "n_window": int(inside.sum()),
        "converged": info["converged"], "t_end": info["t_end"], "n_force_calls": info["n_force_calls"], "max_force": info["max_force"],
        "message": info["message"], "n_snapshots": len(times),
        "n_flipped": n_flipped, "frac_left_window": frac_left,
        "E_before": E0_int + E0_sub, "E_springs_before": E0_int, "E_substrate_before": E0_sub,
        "E_after": E_int + E_sub, "E_springs_after": E_int, "E_substrate_after": E_sub,
        "u_max": u.max(), "u_mean": u.mean(), "u_mean_window": u[inside].mean(),
        "bond_min": length.min(), "bond_mean": length.mean(), "bond_max": length.max(),
        "sigma_min": np.nanmin(sigma_iso[inside]), "sigma_mean": np.nanmean(sigma_iso[inside]), "sigma_max": np.nanmax(sigma_iso[inside]),
    }
    result = {"pos0": pos0, "times": times, "snapshots": snapshots, "pos": pos, "left": left, "sigma_iso": sigma_iso,
              "window": window, "inside": inside, "summary": summary}
    return result
