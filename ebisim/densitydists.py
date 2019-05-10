import numpy as np
import numba as nb
import math as mt

from .physconst import PI, Q_E
from .physconst import MINIMAL_DENSITY, MINIMAL_KBT


@nb.jit
def n_i_boltzmann_s(y, NkT, Z):
    """
    Returns the sum for each CS of the Boltzmann charge distribution of ions [C.m-3].
            the derivative by y of this.

    Input Parameter:
    y - float Potential [V].
    NkT -
    Z -
    """
    N = NkT[:Z + 1]
    kbT = NkT[Z + 1:]
    q_threshold = np.arange(Z + 1)[(N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)]
    rho = 0
    rho_p = 0
    for i in q_threshold:
        rho += N[i] * Q_E * i * mt.exp(-y * i / kbT[i])
        rho_p += - N[i] * Q_E * i ** 2 / kbT[i] * mt.exp(-y * i / kbT[i])
    return rho, rho_p

@nb.jit
def n_i_boltzmann_m(y, NkT, Z):
    """
    Returns the Boltzmann charge distribution for the ions [C.m-3].

    Input Parameter:
    y - np.ndarray Potential [V].
    NkT -
    Z -
    """
    N = NkT[:Z + 1]
    kbT = NkT[Z + 1:]
    q_threshold = np.arange(Z + 1)[(N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)]
    rho_partial = np.zeros((y.size, Z + 2))
    for i in q_threshold:
        rho_partial[:, i] += N[i] * Q_E * i * np.exp(-y * i / kbT[i])
    rho_partial[:, Z + 1] += rho_partial.sum(axis=1)
    return rho_partial


@nb.jit
def n_i_maxwell1_s(y, NkT, Z):
    """
    Returns the Maxwell-Boltzmann charge distribution with 1 degree of freedom the ions.
            the derivative of this distribution regarding y.

            It's wrong!!
    Input Parameters:
    y - np.ndarray Potential [V].
    NkT -
    Z -
    """
    N = NkT[:Z + 1]
    kbT = NkT[Z + 1:]
    q_threshold = np.arange(Z + 1)[(N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)]
    rho = 0
    rho_p = 0
    for i in q_threshold:
        rho += N[i] * Q_E * i * mt.sqrt(y / (kbT[i] * i * PI)) * mt.exp(-y * i / kbT[i])
        rho_p += 0 if y == 0 else\
            -N[i] * Q_E * y * (2 * i * y + kbT[i]) / (2 * PI**1/2 * y ** 2 * kbT[i] / i) * mt.exp(-y * i / kbT[i])
    return rho, rho_p


@nb.jit
def n_i_maxwell1_m(y, NkT, Z):
    """
    Returns the Maxwell-Boltzmann charge distribution with 1 degree of freedom the ions.

    Input Parameters:
    r - Radial position [m].
    y - Potential [V].
    """
    N = NkT[:Z + 1]
    kbT = NkT[Z + 1:]
    q_threshold = np.arange(Z + 1)[(N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)]
    rho_partial = np.zeros((y.size, Z + 2))
    for i in q_threshold:
        rho_partial[:, i] += N[i] * Q_E * i * np.sqrt(y / kbT[i] * i * PI) * np.exp(-y * i / kbT[i])
    rho_partial[:, Z + 1] += rho_partial.sum(axis=1)
    return rho_partial


def n_i_maxwell3(y, NkT, Z):
    """
    Returns the Maxwell-Boltzmann charge distribution with 3 degrees of freedom the ions.
            the derivative of this distribution regarding y.

    Input Parameters:
    r - Radial position [m].
    y - Potential [V].
    """
    N = NkT[:Z + 1]
    kbT = NkT[Z + 1:]
    q_l = np.array(range(Z + 1))
    mask = (N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)
    q_l = q_l[mask]
    rho = sum(2 * N[i] * Q_E * i * (np.clip(y * i / PI, 0, None))**(1/2) *
              (1 / kbT[i])**(1/2) * np.exp(-y * i / kbT[i]) for i in range(len(q_l)))
    rho_p = sum(-2 * N[i] * Q_E * i**2 * (np.clip(y * i / PI, 0, None))**(1/2) *
                (1 / kbT[i])**(5/2) * np.exp(-y * i / kbT[i]) for i in range(len(q_l)))
    return [rho, rho_p]

def n_i_maxwell5(y, NkT, Z):
    """
    Returns the Maxwell-Boltzmann charge distribution with 5 degrees of freedom the ions.
            the derivative of this distribution regarding y.

    Input Parameters:
    r - Radial position [m].
    y - Potential [V].
    """
    N = NkT[:Z + 1]
    kbT = NkT[Z + 1:]
    q_l = np.array(range(Z + 1))
    mask = (N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)
    q_l = q_l[mask]
    rho = sum(4/3 * N[i] * Q_E * i * (1 / PI)**(1/2) * (np.clip(y * i, 0, None))**(3/2)
              * (1 / kbT[i])**(5/2) * np.exp(-y * i / kbT[i]) for i in range(len(q_l)))
    rho_p = sum(-4/3 * N[i] * Q_E * i**2 * (1 / PI)**(1/2) * (np.clip(y * i, 0, None))**(3/2) *
              (1 / kbT[i]) ** (7 / 2) * np.exp(-y * i / kbT[i]) for i in range(len(q_l)))
    return [rho, rho_p]

def n_i_gaussian(y, NkT, Z):
    """
    Returns the Gaussian charge distribution for the ions.
            the derivative of this distribution regarding y.

    Input Parameters
    r - radial position [m]
    y - potential [eV]
    """
    N = NkT[:Z + 1]
    kbT = NkT[Z + 1:]
    q_l = np.array(range(Z + 1))
    mask = (N > MINIMAL_DENSITY) & (kbT > MINIMAL_KBT)
    q_l = q_l[mask]
    sig = kbT
    rho = sum(N[i] * Q_E * i / sig[i] * (2 * PI)**(-1/2) *
              np.exp(-(y / sig[i])**2 / 2) for i in range(len(q_l)))
    rho_p = sum(-N[i] * Q_E * i * y * sig[i]**-3 * (2 * PI)**(-1/2) *
              np.exp(-(y / sig[i])**2 / 2) for i in range(len(q_l)))
    return [rho, rho_p]


@nb.jit
def n_e_normal_s(r, Qe, re):
    """
    Returns the normal charge density distribution of the electron beam.
            the derivative of this distribution regarding y.

    Input Parameter:
    r - float Radial position [m]
    Qe -
    re -
    """
    return -Qe / (PI * re ** 2) if r < re else 0, 0

@nb.jit
def n_e_normal_m(r, Qe, re):
    """
    Returns the normal charge density distribution of the electron beam.

    Input Parameter:
    r - np.ndarray Radial position [m]
    Qe -
    re -
    """
    rho = -Qe / (PI * re ** 2) * np.ones(r.size)
    rho[re < abs(r)] = 0
    return rho


@nb.jit
def n_e_gaussian_s(r, Qe, re):
    """
    Returns the Gaussian charge density distribution of the electron beam.

    Input Parameter:
    r - float Radial position [m]
    Qe -
    re -
    """
    return -Qe / (PI * re ** 2) * mt.exp(-(r / re) ** 2), 0


@nb.jit
def n_e_gaussian_m(r, Qe, re):
    """
    Returns the Gaussian charge density distribution of the electron beam.

    Input Parameter:
    r - np.ndarray Radial position [m]
    Qe -
    re -
    """
    return -Qe / (PI * re ** 2) * np.exp(-(r / re) ** 2)


def n_i_pb(model_n_i=None, no_ions=False, **kwargs):
    """
    Returns the ionic charge density distribution according to model[0].

    Input Parameter:
    model_n_i - string describing the ion distribution
    kwargs for distributions - ('y'=, 'NkT'=, 'Z'=)
    """
    term = r'_m' if isinstance(kwargs['y'], np.ndarray) else '_s'
    return np.zeros(2) if no_ions else globals()['n_i_' + model_n_i + term](**kwargs)


def n_e_pb(model_n_e=None, **kwargs):
    """
    Returns the electron charge density distribution according to model[1].

    Input Parameter:
    model_n_e - string describing the electron distribution
    kwargs for distributions - ('r'=, 'Qe'=, 're'=)
    """
    term = r'_m' if isinstance(kwargs['r'], np.ndarray) else '_s'
    return globals()['n_e_' + model_n_e + term](**kwargs)
