#INFO: Overdamped dynamics, gamma dr_k/dt = F_k(r), for any model; the model only supplies its force function


import numpy as np
from scipy.integrate import solve_ivp


def overdamped(force, pos0, mobile, gamma=1.0, dt_snap=1.0, t_max=1e4, f_tol=1e-6, rtol=1e-10, atol=1e-9):
    """
    Integrates gamma dr_k/dt = F_k for the mobile particles (the pinned ones stay at pos0) until every |F_k| < f_tol or t = t_max.

    Parameters
    ----------
    force: Function force(pos) returning an array of shape (N, 2), the total force on each particle
    pos0: Array such that pos0[k] gives the STARTING (x, y) coordinates of particle k
    mobile: Boolean array, mobile[k] is True if particle k is allowed to move
    gamma: Damping coefficient (one value for every particle)
    dt_snap: Time between saved snapshots
    t_max: Stop at this time even if the forces are not yet below f_tol
    f_tol: Stop once every mobile particle has |F_k| < f_tol (a local energy minimum has been reached)
    rtol: Relative error tolerance of the adaptive integrator; keep tiny, since it multiplies the absolute coordinates (~10-100)
    atol: Absolute error tolerance of the adaptive integrator (in units of length); this is the one that controls the accuracy

    Returns
    ----------
    times: Array of shape (S,), the time of each snapshot (0, dt_snap, 2 dt_snap, ..., and the stopping time last)
    snapshots: Array of shape (S, N, 2), snapshots[n] gives the positions at times[n]
    info: Dictionary with converged (stopped by f_tol), t_end, max_force (largest |F_k| over mobile particles at the end), n_force_calls, message
    """
    pos = pos0.astype(float).copy() # pinned particles keep these coordinates for the whole run

    #INFO: solve_ivp needs one flat vector of unknowns; x = (x_0, y_0, x_1, y_1, ...) over the mobile particles only
    def unpack(x):
        P = pos.copy()
        P[mobile] = x.reshape(-1, 2)
        return P

    def max_force(P):
        F = force(P)[mobile]
        return np.sqrt(np.max(np.sum(F**2, axis=1)))

    # WARN: the equation of motion. Overdamped: no inertia, the velocity is the force divided by gamma, so the particles slide downhill
    # in the energy (E never increases). gamma only sets the time unit while every particle has the same gamma.
    def rhs(t, x):
        return force(unpack(x))[mobile].ravel() / gamma
    # WARN: end

    # WARN: the stopping rule. The event is max|F_k| - f_tol; solve_ivp stops at the first time it crosses zero (terminal = True).
    # The integrator picks its own step sizes to keep each step's error below atol + rtol |x|. x holds absolute coordinates, so a
    # "normal" rtol like 1e-6 would allow ~1e-5 errors on a patch of size ~10 and the run would jitter at max|F| ~ 1e-5, never reaching
    # f_tol; hence rtol is tiny and atol sets the accuracy (needs atol * stiffness << f_tol). Snapshots are taken every dt_snap by integrating
    # one dt_snap at a time and restarting from the last state, which does not change the trajectory.
    def stop(t, x):
        return max_force(unpack(x)) - f_tol
    stop.terminal = True
    stop.direction = -1 # only when the force drops below f_tol
    # WARN: end

    times = [0.0]
    snapshots = [pos.copy()]
    x = pos[mobile].ravel()
    t = 0.0
    n_force_calls = 0
    converged = max_force(pos) < f_tol # already at a minimum: nothing to do
    message = "already below f_tol at t = 0" if converged else "reached t_max"

    while not converged and t < t_max:
        t_next = min(t + dt_snap, t_max)
        sol = solve_ivp(rhs, (t, t_next), x, method="RK45", rtol=rtol, atol=atol, events=stop)
        n_force_calls += sol.nfev
        if sol.status == -1: # integration failed
            message = sol.message
            break
        x = sol.y[:, -1]
        t = sol.t[-1]
        times.append(t)
        snapshots.append(unpack(x))
        if sol.status == 1: # the stop event fired
            converged = True
            message = "max|F| below f_tol"

    P = snapshots[-1]
    info = {"converged": converged, "t_end": t, "max_force": max_force(P), "n_force_calls": n_force_calls, "message": message}
    return np.array(times), np.array(snapshots), info
