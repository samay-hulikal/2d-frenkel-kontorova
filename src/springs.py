#INFO: Harmonic springs between neighbouring cell centres; the springs try to keep the bees' natural cell spacing a


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
    dx: Array of shape (M,), x component of r_q - r_p for every bond l
    dy: Array of shape (M,), y component of r_q - r_p for every bond l
    length: Array of shape (M,), |r_q - r_p| for every bond l
    """
    p = bonds[:, 0] # first particle of every bond, shape (M,)
    q = bonds[:, 1] # second particle of every bond, shape (M,)

    dx = pos[q, 0] - pos[p, 0] # x component of the bond vector, pointing from p to q
    dy = pos[q, 1] - pos[p, 1] # y component of the bond vector, pointing from p to q
    length = np.sqrt(dx**2 + dy**2)
    return dx, dy, length


def spring_energy(pos, bonds, k=1.0, a=1.0):
    """
    Total spring energy, E = (k / 2) sum_bonds (|r_q - r_p| - a)^2.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)

    Returns
    ----------
    E: Total spring energy (a single number)
    """
    dx, dy, length = bond_vectors(pos, bonds)
    stretch = length - a # > 0 if the bond is stretched, < 0 if it is compressed
    E = (k / 2) * np.sum(stretch**2)
    return E


def spring_force(pos, bonds, k=1.0, a=1.0):
    """
    Spring force on each particle, F = -grad E.

    Parameters
    ----------
    pos: Array such that pos[k] gives the (x, y) coordinates of particle k
    bonds: Array such that bonds[l] = [p, q] means particles p and q are connected by bond number l
    k: Spring constant
    a: Rest length of the springs (natural cell spacing)

    Returns
    ----------
    F: Array of shape (N, 2) such that F[k] is the (x, y) spring force on particle k
    """
    dx, dy, length = bond_vectors(pos, bonds)
    stretch = length - a

    # WARN: force of one bond. dE/dr_q = k (|d| - a) d/|d| with d = r_q - r_p, so the force on q is
    # -k (|d| - a) d/|d| (pulls q towards p if stretched, pushes it away if compressed) and the force on p is the opposite.
    fx = -k * stretch * dx / length # x force on q from bond l, shape (M,)
    fy = -k * stretch * dy / length # y force on q from bond l, shape (M,)
    # WARN: end

    # WARN: summing over bonds. A particle belongs to up to 6 bonds, so its force is the sum over all of them.
    # np.add.at(Fx, q, fx) does Fx[q[l]] += fx[l] for every l, adding repeated indices correctly (Fx[q] += fx would not).
    N = pos.shape[0]
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    np.add.at(Fx, bonds[:, 1], fx) # force on q
    np.add.at(Fy, bonds[:, 1], fy)
    np.add.at(Fx, bonds[:, 0], -fx) # equal and opposite force on p
    np.add.at(Fy, bonds[:, 0], -fy)
    # WARN: end

    F = np.column_stack([Fx, Fy]) # F[k] = (Fx_k, Fy_k)
    return F
