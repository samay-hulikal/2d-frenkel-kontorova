#INFO: Underlying bumpy surface representing the lattice of 3D printed foundation; the wax coating on the foundations acts as the potential trying to pull the bees' lattice to conform to the 3D printed foundation


import numpy as np

def reciprocal_vectors(a_s=1.0):
    """
    The three shortest reciprocal-lattice vectors of a triangular substrate with spacing a_s.

    Parameters
    ----------
    a_s: Spacing between neighboring substrate minima

    Returns
    ----------
    G: Array of shape (3, 2) such that G[i] is the (x, y) components of the reciprocal vector G_i
    """

    #INFO: The substrate lattice has primitive vectors v_1 = a_s*(1, 0) and v_2 = a_s*(1/2, np.sqrt(3)/2), the same orientation as triangular_patch, with a minimum at the origin; v_i . G_j = delta_ij
    G1 = (2 / (np.sqrt(3) * a_s)) * np.array([np.sqrt(3) / 2, - 1 / 2])
    G2 = (2 / (np.sqrt(3) * a_s)) * np.array([0, 1])
    G3 = - (G1 + G2) # The midpoints of the walls behave differently without this vector
    G = np.array([G1, G2, G3])
    return G


def substrate_energy(pos, V0=1.0, a_s=1.0):
    """
    Substrate energy of each particle, V(r) = (2 V0 / 9)(3 - cos(2 * np.pi * G_1 . r) - cos(2 * np.pi * G_2 . r) - cos(2 * np.pi * G_3 . r)).

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    V0: Peak height of the potential-- V = 0 at the minima (cell centers), V = V0 at the maxima (3-vertices), V = 8V0/9 at the saddles (wall midpoints)
    a_s: Spacing between neighboring substrate minima

    Returns
    ----------
    V: V[k] is the substrate energy of particle k
    """
    #INFO: Having cell center at the 3-vertices of the foundation cost the highest energy, having cell centers at the centers of the foundation cost the least energy, having the cell centers at the wall-midpoints fall in between these two costs

    G1, G2, G3 = reciprocal_vectors(a_s) # each is a length-2 array (x, y)
    x = pos[:, 0] # x coordinates of all particles, shape (N,)
    y = pos[:, 1] # y coordinates of all particles, shape (N,)

    phase1 = 2 * np.pi * (G1[0] * x + G1[1] * y) # 2π G_1 . r_k for every particle k
    phase2 = 2 * np.pi * (G2[0] * x + G2[1] * y) # 2π G_2 . r_k for every particle k
    phase3 = 2 * np.pi * (G3[0] * x + G3[1] * y) # 2π G_3 . r_k for every particle k

    V = (2 * V0 / 9) * (3 - np.cos(phase1) - np.cos(phase2) - np.cos(phase3))
    return V


def substrate_force(pos, V0=1.0, a_s=1.0):
    """
    Substrate force on each particle, F = -grad V = -(2 V0 / 9) * 2 * np.pi * sum_i G_i sin(2 * np.pi * G_i . r).

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    V0: Peak height of the potential
    a_s: Spacing between neighboring substrate minima

    Returns
    ----------
    F: Array of shape (N, 2) such that F[k] is the (x, y) substrate force on particle k
    """
    G1, G2, G3 = reciprocal_vectors(a_s) # each is a length-2 array (x, y)
    x = pos[:, 0] # x coordinates of all particles, shape (N,)
    y = pos[:, 1] # y coordinates of all particles, shape (N,)

    phase1 = 2 * np.pi * (G1[0] * x + G1[1] * y) # 2π G_1 . r_k for every particle k
    phase2 = 2 * np.pi * (G2[0] * x + G2[1] * y) # 2π G_2 . r_k for every particle k
    phase3 = 2 * np.pi * (G3[0] * x + G3[1] * y) # 2π G_3 . r_k for every particle k

    #INFO: Calculated such that F[k] is the force on particle k
    Fx = -(2 * V0 / 9) * 2 * np.pi * (G1[0] * np.sin(phase1) + G2[0] * np.sin(phase2) + G3[0] * np.sin(phase3))
    Fy = -(2 * V0 / 9) * 2 * np.pi * (G1[1] * np.sin(phase1) + G2[1] * np.sin(phase2) + G3[1] * np.sin(phase3))

    F = np.column_stack([Fx, Fy]) # F[k] = (Fx_k, Fy_k)
    return F
