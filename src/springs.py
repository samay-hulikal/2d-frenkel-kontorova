#INFO: Harmonic springs between neighboring cell centers; the springs try to keep the bees' natural cell spacing a


import numpy as np


def bond_vectors(pos, bonds):
    """
    Vector and length of every bond.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l

    Returns
    ----------
    dx: Array of shape (M,), x component of r_q - r_p for every bond l; dx[l] gives the x-component of the vector for bond l
    dy: Array of shape (M,), y component of r_q - r_p for every bond l; dy[l] gives the y-component of the vector for bond l
    length: Array of shape (M,), |r_q - r_p| for every bond l; length[l] gives the length of bond l
    """
    p = bonds[:, 0] # first particle of every bond, shape (M,)
    q = bonds[:, 1] # second particle of every bond, shape (M,)

    #INFO: Note the direction of the bond vector-- it points from p to q
    dx = pos[q, 0] - pos[p, 0] # x component of the bond vector
    dy = pos[q, 1] - pos[p, 1] # y component of the bond vector
    length = np.sqrt(dx**2 + dy**2)
    return dx, dy, length


def spring_energy(pos, bonds, k=1.0, a=1.0):
    """
    Total spring energy, V_int = (k / 2) sum_bonds (|r_q - r_p| - a)^2.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)

    Returns
    ----------
    V_int: Total spring energy (a single number)
    """
    dx, dy, length = bond_vectors(pos, bonds)

    dl = length - a

    #INFO: Total spring energy
    V_int = (k / 2) * np.sum(dl**2)
    return V_int


def spring_force(pos, bonds, k=1.0, a=1.0):
    """
    Spring force on each particle, F_int = -grad V_int.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)

    Returns
    ----------
    F_int: Array of shape (N, 2) such that F_int[k] is the (x, y) spring force on particle k
    """
    dx, dy, length = bond_vectors(pos, bonds)
    dl = length - a

    #INFO: Force exerted by bond l on particle p (the tail of the bond vector); this is where the direction of the bond vector matters
    fx = + k * dl * dx / length # x force on p from bond l
    fy = + k * dl * dy / length # y force on p from bond l

    #INFO: Adding the forces from all the bonds that contain a given particle
    N = pos.shape[0]
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    np.add.at(Fx, bonds[:, 0], fx) # forces on p
    np.add.at(Fy, bonds[:, 0], fy)

    np.add.at(Fx, bonds[:, 1], -fx) # equal and opposite forces on q
    np.add.at(Fy, bonds[:, 1], -fy)

    F_int = np.column_stack([Fx, Fy]) # F[k] = (Fx_k, Fy_k)
    return F_int
