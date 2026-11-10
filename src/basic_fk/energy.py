#INFO: Total energy and force of the basic FK model: springs + substrate

#INFO: The simulation is currently NOT dimensionless; k and V0 must be chosen carefully

import numpy as np

from src.basic_fk.springs import spring_energy, spring_force
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
