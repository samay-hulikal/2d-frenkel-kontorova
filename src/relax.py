#INFO: Relaxing the lattice to a local energy minimum; springs + substrate, only the mobile particles move

#INFO: The simulation is currently NOT dimensionless; k and V0 must be chosen carefully

import numpy as np
from scipy.optimize import minimize

from src.springs import spring_energy, spring_force
from src.substrate import substrate_energy, substrate_force


def total_energy(pos, bonds, k=1.0, a=1.0, V0=1.0, a_s=1.0):
    """
    Total energy, E = V_int + sum_k V_sub[k].

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)
    V0: Peak height of the substrate potential
    a_s: Spacing between neighboring substrate minima

    Returns
    ----------
    E: Total energy (a single number)
    """
    V_int = spring_energy(pos, bonds, k, a) # already summed over bonds
    V_sub = np.sum(substrate_energy(pos, V0, a_s)) # substrate_energy gives one value per particle, so sum it
    E = V_int + V_sub
    return E


def total_force(pos, bonds, k=1.0, a=1.0, V0=1.0, a_s=1.0):
    """
    Total force on each particle, F = F_int + F_sub = -grad E.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)
    V0: Peak height of the substrate potential
    a_s: Spacing between neighboring substrate minima

    Returns
    ----------
    F: Array of shape (N, 2) such that F[k] is the (x, y) total force on particle k
    """

    #INFO: Force on particle k as a vector
    F = spring_force(pos, bonds, k, a) + substrate_force(pos, V0, a_s)
    return F


def relax(pos0, bonds, mobile, k=1.0, a=1.0, V0=1.0, a_s=1.0, f_tol=1e-6, maxiter=10000):
    """
    Moves the mobile particles to a local minimum of the total energy using L-BFGS; the pinned particles stay at pos0.

    Parameters
    ----------
    pos0: Array such that pos0[k] gives the STARTING (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    mobile: Boolean array, mobile[k] is True if particle k is allowed to move
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)
    V0: Peak height of the substrate potential
    a_s: Spacing between neighboring substrate minima
    f_tol: Stop once every mobile particle has |F_k| < f_tol; tolerance to indicate that a local minimum has been reached
    maxiter: Maximum number of L-BFGS iterations

    Returns
    ----------
    pos: Array of shape (N, 2), the relaxed positions
    info: Dictionary with energy, max_force (largest |F_k| over mobile particles), n_iter, converged, message
    """

    #INFO: L-BFGS needs one flat vector of unknowns; x = (x_0, y_0, x_1, y_1, ...) over the mobile particles only
    x0 = pos0[mobile].ravel()

    def positions(x):
        pos = pos0.copy() # pinned particles keep their starting positions

        #INFO: Rebuilds the flat vector into a form that the energy functions can use
        pos[mobile] = x.reshape(-1, 2)
        return pos

    #INFO: The gradient of E with respect to the mobile coordinates is minus the total force on the mobile particles, flattened in the same (x_0, y_0, x_1, y_1, ...) order as x
    def energy_and_gradient(x):
        pos = positions(x)
        E = total_energy(pos, bonds, k, a, V0, a_s)
        grad = -total_force(pos, bonds, k, a, V0, a_s)[mobile].ravel() # Force is - grad
        return E, grad


    #INFO: Stopping rules for the minimization: 
    # Starts the particles at x0, uses method L-BFGS.
    # scipy's gtol stops when the largest single gradient COMPONENT is below gtol; since |F_k| <= sqrt(2) * max(|F_kx|, |F_ky|), gtol = f_tol / sqrt(2) guarantees |F_k| < f_tol
    # ftol = 0 switches off scipy's other stop (small relative energy change), which can quit while forces are still large
    # maxfun is raised because one iteration can take several energy evaluations (line search)
    result = minimize(energy_and_gradient, x0, jac=True, method="L-BFGS-B",
                      options={"gtol": f_tol / np.sqrt(2), "ftol": 0.0, "maxiter": maxiter, "maxfun": 20 * maxiter})

    #INFO: Checking convergence ourselves instead of trusting scipy's flag
    pos = positions(result.x)
    F = total_force(pos, bonds, k, a, V0, a_s)[mobile]
    max_force = np.max(np.sqrt(F[:, 0]**2 + F[:, 1]**2)) # largest |F_k| over the mobile particles

    info = {
        "energy": total_energy(pos, bonds, k, a, V0, a_s),
        "max_force": max_force,
        "n_iter": result.nit,
        "converged": bool(max_force < f_tol),
        "message": result.message,
    }
    return pos, info


def relax_ramped(pos0, bonds, mobile, k=1.0, a=1.0, V0=1.0, a_s=1.0, a_start=1.0, n_stages=20, f_tol=1e-6, maxiter=10000):
    """
    Relaxes in stages, moving the springs' rest length from a_start to a; avoids the patch folding over itself when it starts strongly strained.

    Parameters
    ----------
    pos0: Array such that pos0[k] gives the STARTING (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    mobile: Boolean array, mobile[k] is True if particle k is allowed to move
    k: Spring constant
    a: Final rest length of the springs (natural cell spacing)
    V0: Peak height of the substrate potential
    a_s: Spacing between neighboring substrate minima
    a_start: Rest length used in the first stage (normally the spacing of the starting patch, so the springs start relaxed)
    n_stages: Number of stages; each one is a full relax
    f_tol: Stop each stage once every mobile particle has |F_k| < f_tol
    maxiter: Maximum number of L-BFGS iterations per stage

    Returns
    ----------
    pos: Array of shape (N, 2), the relaxed positions at rest length a
    info: Dictionary as from relax() for the last stage, but n_iter is the total over all stages
    """
    # WARN: ramping the rest length. Stage n relaxes with rest length a_n, a_n going in equal steps from a_start (exclusive) to a (inclusive),
    # each stage starting from the previous stage's result. Only the last stage uses the real model (rest length a); the earlier stages
    # only choose the path, so the patch shrinks or grows gently instead of snapping. The result is still a local minimum of the real model.
    pos = pos0
    n_iter = 0
    for a_n in np.linspace(a_start, a, n_stages + 1)[1:]:
        pos, info = relax(pos, bonds, mobile, k, a_n, V0, a_s, f_tol, maxiter)
        n_iter += info["n_iter"]
    # WARN: end
    info["n_iter"] = n_iter
    return pos, info
