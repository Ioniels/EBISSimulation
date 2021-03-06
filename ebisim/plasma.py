"""
This module contains functions for calculating collission rates and related plasma parameters
"""

import math
import numba
import numpy as np

from .physconst import M_E, M_P, PI, EPS_0, Q_E, C_L, M_E_EV
from .physconst import MINIMAL_DENSITY

@numba.jit
def electron_velocity(e_kin):
    """
    Returns the electron velocity corresponding to a kin. energy in m/s

    Input Parameters
    e_kin - electron energy in eV
    """
    return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)

@numba.jit
def clog_ei(Ni, Ne, kbTi, kbTe, Ai, qi):
    """
    The coulomb logarithm for ion electron collisions
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    kbTe - electron energy /temperature in eV
    Ai - ion mass in amu
    qi - ion charge
    """
    Ni *= 1e-6 # go from 1/m**3 to 1/cm**3
    Ne *= 1e-6
    Mi = Ai * M_P
    if   qi*qi*10 >= kbTe >= kbTi * M_E / Mi:
        return 23 - math.log(Ne**0.5 * qi * kbTe**-1.5)
    elif kbTe >= qi*qi*10 >= kbTi * M_E / Mi:
        return 24 - math.log(Ne**0.5 / kbTe)
    elif kbTe <= kbTi * M_E / Mi:
        return 16 - math.log(Ni**0.5 * kbTi**-1.5 * qi * qi * Ai)
    # The next case should not usually arise in any realistic situation but the solver may probe it
    # Hence it is purely a rough guess
    elif qi*qi*10 <= kbTi * M_E / Mi <= kbTe:
        return 24 - math.log(Ne**0.5 / kbTe)

# @numba.jit
# def clog_ei_vec(Ni, Ne, kbTi, kbTe, Ai):
#     """
#     Vector version of clog_ei
#     Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
#     Ni - ion density in 1/m^3
#     Ne - electron density in 1/m^3
#     kbTi - electron energy /temperature in eV
#     kbTe - electron energy /temperature in eV
#     Ai - ion mass in amu
#     """
#     n = Ni.size
#     clog = np.zeros(n)
#     for q in range(1, n):
#         clog[q] = clog_ei(Ni[q], Ne, kbTi[q], kbTe, Ai, q)
#     return clog

@numba.jit
def clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    """
    The coulomb logarithm for ion ion collisions
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Ai/Aj - ion mass in amu
    qi/qj - ion charge
    """
    Ni *= 1e-6 # go from 1/m**3 to 1/cm**3
    Nj *= 1e-6
    A = qi * qj * (Ai + Aj) / (Ai * kbTj + Aj * kbTi)
    B = Ni * qi * qi / kbTi + Nj * qj * qj / kbTj
    clog = 23 - math.log(A * B**0.5)
    if clog < 0:
        clog = 0
    return clog

# @numba.jit
# def clog_ii_mat(Ni, Nj, kbTi, kbTj, Ai, Aj):
#     """
#     Matrix version of clog_ii
#     Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
#     The coulomb logarithm for ion ion collisions
#     Ni/Nj - ion density in 1/m^3
#     kbTi/kbTj - electron energy /temperature in eV
#     Ai/Aj - ion mass in amu
#     qi/qj - ion charge
#     """
#     ni = Ni.size
#     nj = Nj.size
#     clog = np.zeros((ni, nj))
#     for qj in range(1, nj):
#         for qi in range(1, ni):
#             clog[qi, qj] = clog_ii(Ni[qi], Nj[qj], kbTi[qi], kbTj[qj], Ai, Aj, qi, qj)
#     return clog

@numba.jit
def coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi):
    """
    Computes the coulomb cross section for electron ion elastic collisions
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    Ee - electron kinetic energy in eV
    Ai - ion mass in amu
    qi - ion charge
    """
    if qi == 0:
        return 0
    v_e = electron_velocity(Ee)
    clog = clog_ei(Ni, Ne, kbTi, Ee, Ai, qi)
    return 4 * PI * (qi * Q_E * Q_E / (4 * PI * EPS_0 * M_E))**2 * clog / v_e**4

# @numba.jit
# def coulomb_xs_vec(Ni, Ne, kbTi, Ee, Ai):
#     """
#     Vector version of coulomb_xs
#     Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
#     Ni - ion density in 1/m^3
#     Ne - electron density in 1/m^3
#     kbTi - electron energy /temperature in eV
#     Ee - electron kinetic energy in eV
#     Ai - ion mass in amu
#     """
#     n = Ni.size
#     xs = np.zeros(n)
#     for q in range(1, n):
#         xs[q] = coulomb_xs(Ni[q], Ne, kbTi[q], Ee, Ai, q)
#     return xs

@numba.jit
def ion_coll_rate(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    """
    Collision rate of ions species "i" for target "j"
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Ai/Aj - ion mass in amu
    qi/qj - ion charge
    """
    # Artifically clamp collision rate to zero if either density is very small
    # This is a reasonable assumption and prevents instabilities when calling the coulomb logarithm
    if Ni < MINIMAL_DENSITY or Nj < MINIMAL_DENSITY or kbTi < 0 or kbTj < 0:
        return 0
    clog = clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj)
    kbTi_SI = kbTi * Q_E
    Mi = Ai * M_P
    const = 4 / 3 / (4 * PI * EPS_0)**2 * math.sqrt(2 * PI)
    return const * Nj * (qi * qj * Q_E * Q_E / Mi)**2 * (Mi/kbTi_SI)**1.5 * clog

@numba.jit
def ion_coll_rate_mat(Ni, Nj, kbTi, kbTj, Ai, Aj):
    """
    Matric of ollision rates of ions species "i" for target "j"
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Ai/Aj - ion mass in amu
    """
    ni = Ni.size
    nj = Nj.size
    r_ij = np.zeros((ni, nj))
    for qi in range(1, ni):
        for qj in range(1, nj):
            r_ij[qi, qj] = ion_coll_rate(Ni[qi], Nj[qj], kbTi[qi], kbTj[qj], Ai, Aj, qi, qj)
    return r_ij

@numba.jit
def electron_heating_vec(Ni, Ne, kbTi, Ee, Ai):
    """
    Computes the heating rate due to elastic electron ion collisions
    Referred to as Spitzer Heating
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    Ee - electron kinetic energy in eV
    Ai - ion mass in amu
    """
    n = Ni.size
    heat = np.zeros(n)
    const = Ne * electron_velocity(Ee) * 2 * M_E / (Ai * M_P) * Ee
    for qi in range(1, n):
        if Ni[qi] > MINIMAL_DENSITY:
            heat[qi] = const * Ni[qi] * coulomb_xs(Ni[qi], Ne, kbTi[qi], Ee, Ai, qi)
    return heat

@numba.jit
def energy_transfer_vec(Ni, Nj, kbTi, kbTj, Ai, Aj, rij):
    """
    The energy transfer term for species "i" (with respect to species "j")
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Ai/Aj - ion mass in amu
    qi/qj - ion charge
    rij - ion ion collision rates in 1/s
    """
    ni = Ni.size
    nj = Nj.size
    trans_i = np.zeros(ni)
    for qi in range(1, ni):
        for qj in range(1, nj):
            if kbTi[qi] < 0 or kbTj[qj] < 0:
                trans_i[qi] += 0
            else:
                trans_i[qi] += 2 * rij[qi, qj] * Ni[qi] * Ai/Aj * (kbTj[qj] - kbTi[qi]) / \
                               (1 + (Ai * kbTj[qj]) / (Aj * kbTi[qi]))**1.5
    return trans_i

@numba.jit
def loss_frequency_axial(kbTi, V):
    """
    Axial ion loss frequency
    kbTi - electron energy /temperature in eV
    V - Trap depth in V
    """
    valid = kbTi > 0
    q = np.arange(kbTi.size, dtype=float)
    w = q * V
    w[valid] = w[valid] / kbTi[valid]
    w[np.logical_not(valid)] = np.inf
    w[0] = np.inf # fake value for neutrals -> essentially infinite trap
    return w

@numba.jit
def loss_frequency_radial(kbTi, Ai, V, B, r_dt):
    """
    Radial ion loss frequency
    kbTi - electron energy /temperature in eV
    Ai - ion mass in amu
    V - Trap depth in V
    B - Axial magnetic field in T
    r_dt - drift tube in m
    """
    valid = kbTi > 0
    q = np.arange(kbTi.size, dtype=float)
    w = q
    w[valid] = w[valid] * (V + B * r_dt * np.sqrt(2 * kbTi[valid] * Q_E /(3*M_P*Ai))) / kbTi[valid]
    w[np.logical_not(valid)] = np.inf
    w[0] = np.inf # fake value for neutrals -> essentially infinite trap
    return w

@numba.jit
def escape_rate_axial(Ni, kbTi, ri, V):
    """
    Axial escape rate
    Ni - ion density in 1/m^3
    kbTi - electron energy /temperature in eV
    ri - cumulated ion ion collision rates in 1/s
    V - Trap depth in V
    """
    w = loss_frequency_axial(kbTi, V)
    return escape_rate(Ni, ri, w)

@numba.jit
def escape_rate_radial(Ni, kbTi, ri, Ai, V, B, r_dt):
    """
    Radial escape rate
    Ni - ion density in 1/m^3
    kbTi - electron energy /temperature in eV
    ri - cumulated ion ion collision rates in 1/s
    Ai - ion mass in amu
    V - Trap depth in V
    B - Axial magnetic field in T
    r_dt - drift tube in m
    """
    w = loss_frequency_radial(kbTi, Ai, V, B, r_dt)
    return escape_rate(Ni, ri, w)

@numba.jit
def escape_rate(Ni, ri, w):
    """
    Generic escape rate - to be called by axial and radial escape
    Ni - ion density in 1/m^3
    ri - cumulated ion ion collision rates in 1/s
    w - loss frequency in 1/s
    """
    esc = 3 / math.sqrt(2) * Ni * ri * np.exp(-w) / w
    esc[esc < 0] = 0 # this cleans neutrals and any other faulty stuff
    esc[Ni < MINIMAL_DENSITY] = 0
    return esc
