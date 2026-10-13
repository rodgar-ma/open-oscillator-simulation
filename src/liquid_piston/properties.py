"""Water and steam properties used by the 0D model.

Saturation pressure and temperature follow IAPWS-IF97 region 4 (the
industrial standard formulation). Everything else is a deliberately simple,
internally consistent model:

* liquid: incompressible, constant specific heat ``C_L``
* vapour: ideal gas with constant ``CP_V``

Both enthalpies share one reference, chosen so that the latent heat is exact
at 100 degC. Because ``CP_V - C_L`` is close to the real slope of h_fg(T),
the latent heat stays within 1 % of steam-table values between 7 and
127 degC (checked against CoolProp in ``tests/test_properties.py``). Using one consistent
reference is what makes the energy balances of the model close exactly.
"""

from __future__ import annotations

import numpy as np

# --- constants (SI) ---------------------------------------------------------
R_V = 461.526          # specific gas constant of water vapour, J/(kg K)
CP_V = 1890.0          # ideal-gas cp of steam around 100 degC, J/(kg K)
CV_V = CP_V - R_V      # J/(kg K)
GAMMA_V = CP_V / CV_V  # ~1.32
C_L = 4186.0           # liquid water specific heat, J/(kg K)

_T_REF = 373.15        # K
_HFG_REF = 2256.4e3    # latent heat at 100 degC, J/kg
H0_V = _HFG_REF - (CP_V - C_L) * _T_REF  # vapour enthalpy offset, J/kg

# --- IAPWS-IF97 region 4 coefficients ----------------------------------------
_N = (
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
    0.65017534844798e3,
)

T_MIN, T_MAX = 273.15, 647.096       # K, validity range of region 4
P_MIN, P_MAX = 611.213, 22.064e6     # Pa


def p_sat(T):
    """Saturation pressure in Pa for temperature ``T`` in K (IAPWS-IF97)."""
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = _N
    T = np.clip(np.asarray(T, dtype=float), T_MIN, T_MAX)
    th = T + n9 / (T - n10)
    A = th**2 + n1 * th + n2
    B = n3 * th**2 + n4 * th + n5
    C = n6 * th**2 + n7 * th + n8
    p_mpa = (2.0 * C / (-B + np.sqrt(B**2 - 4.0 * A * C))) ** 4
    return p_mpa * 1e6


def T_sat(p):
    """Saturation temperature in K for pressure ``p`` in Pa (IAPWS-IF97)."""
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = _N
    p = np.clip(np.asarray(p, dtype=float), P_MIN, P_MAX)
    beta = (p / 1e6) ** 0.25
    E = beta**2 + n3 * beta + n6
    F = n1 * beta**2 + n4 * beta + n7
    G = n2 * beta**2 + n5 * beta + n8
    D = 2.0 * G / (-F - np.sqrt(F**2 - 4.0 * E * G))
    return 0.5 * (n10 + D - np.sqrt((n10 + D) ** 2 - 4.0 * (n9 + n10 * D)))


# --- consistent caloric model -------------------------------------------------
def h_liquid(T):
    """Specific enthalpy of liquid water, J/kg."""
    return C_L * np.asarray(T, dtype=float)


def h_vapour(T):
    """Specific enthalpy of (ideal-gas) vapour, J/kg, same reference as liquid."""
    return CP_V * np.asarray(T, dtype=float) + H0_V


def u_vapour(T):
    """Specific internal energy of vapour, J/kg."""
    return CV_V * np.asarray(T, dtype=float) + H0_V


def h_fg(T):
    """Latent heat of vaporisation, J/kg."""
    return h_vapour(T) - h_liquid(T)


def s_liquid(T, T0=273.15):
    """Specific entropy of liquid, J/(kg K), relative to liquid at ``T0``."""
    return C_L * np.log(np.asarray(T, dtype=float) / T0)


def s_vapour(T, p, T0=273.15, T_anchor=373.15):
    """Specific entropy of vapour, J/(kg K), same reference as ``s_liquid``.

    The vapour is anchored to saturated vapour at ``T_anchor``
    (s_g = s_f + h_fg / T) and extended as an ideal gas.
    """
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)
    s_g_anchor = s_liquid(T_anchor, T0) + h_fg(T_anchor) / T_anchor
    return (s_g_anchor + CP_V * np.log(T / T_anchor)
            - R_V * np.log(p / p_sat(T_anchor)))


def vapour_density(p, T):
    """Ideal-gas vapour density, kg/m^3."""
    return np.asarray(p, dtype=float) / (R_V * np.asarray(T, dtype=float))
