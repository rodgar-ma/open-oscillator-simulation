"""Compressible flow through the ejection (check) valve.

Isentropic nozzle flow of an ideal gas through an orifice of area ``area``
with discharge coefficient ``cd``. The flow is choked when the pressure
ratio falls below the critical ratio. The valve is a check valve: there is no
reverse flow when the chamber pressure is below the outlet pressure.
"""

from __future__ import annotations

import numpy as np


def critical_pressure_ratio(gamma: float) -> float:
    """Downstream/upstream pressure ratio at which the flow chokes."""
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def orifice_mass_flow(p_up, T_up, p_down, area, cd, R, gamma):
    """Mass flow (kg/s) from upstream stagnation state (p_up, T_up) to p_down.

    Returns 0 when ``p_up <= p_down`` (check valve closed).
    """
    if area <= 0.0 or p_up <= p_down:
        return 0.0
    r = max(p_down / p_up, critical_pressure_ratio(gamma))
    psi = r ** (2.0 / gamma) - r ** ((gamma + 1.0) / gamma)
    return cd * area * p_up * np.sqrt(2.0 * gamma / ((gamma - 1.0) * R * T_up) * psi)


def choked_mass_flow(p_up, T_up, area, cd, R, gamma):
    """Closed-form choked mass flow, used as an independent check in tests."""
    return (cd * area * p_up * np.sqrt(gamma / (R * T_up))
            * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))
