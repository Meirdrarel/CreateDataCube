import numpy as np
from scipy.special import i0, i1, k0, k1


def exponential_velocity(r, rt, vmax):
    """
    Velocity function for an exponential profile

    :param ndarray r: 2D array which contain the radius
    :param int rt: radius at which the maximum velocity is reached
    :param float vmax: Maximum velocity of the model
    """

    rd = rt / 2.15        # disk scale length
    vr = np.zeros(np.shape(r))
    q = np.where(r != 0)      # To prevent any problem in the center

    vr[q] = r[q] / rd * vmax / 0.88 * np.sqrt(i0(0.5 * r[q] / rd) * k0(0.5 * r[q] / rd) - i1(0.5 * r[q] / rd) * k1(0.5 * r[q] / rd))

    return vr


def flat_velocity(r, rt, vmax):
    """
    Velocity function for flat profile

    :param ndarray r: 2D array which contain the radius
    :param int rt: radius at which the maximum velocity is reached
    :param float vmax: Maximum velocity of the model
    """

    vr = np.zeros(np.shape(r))

    vr[np.where(r <= rt)] = vmax*r[np.where(r <= rt)]/rt
    vr[np.where(r > rt)] = vmax

    return vr


def arctan_velocity(r, rt, vmax):
    """
    Velocity function for an arctan profile

    :param ndarray r: 2D array which contain the radius
    :param int rt: radius at which the maximum velocity is reached
    :param float vmax: Maximum velocity of the model
    """

    return 2*vmax/np.pi*np.arctan(2*r/rt)

