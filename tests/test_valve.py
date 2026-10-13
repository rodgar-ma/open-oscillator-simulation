"""Ejection valve: compressible orifice flow limits."""

import numpy as np
import pytest

from liquid_piston import properties as wp
from liquid_piston.valve import choked_mass_flow, critical_pressure_ratio, orifice_mass_flow

ARGS = dict(area=2e-3, cd=0.8, R=wp.R_V, gamma=wp.GAMMA_V)


def test_closed_for_reverse_flow():
    assert orifice_mass_flow(100e3, 400.0, 155e3, **ARGS) == 0.0
    assert orifice_mass_flow(155e3, 400.0, 155e3, **ARGS) == 0.0


def test_choked_matches_closed_form():
    m = orifice_mass_flow(500e3, 450.0, 100e3, **ARGS)
    assert m == pytest.approx(choked_mass_flow(500e3, 450.0, **ARGS), rel=1e-12)


def test_continuous_at_critical_ratio():
    rc = critical_pressure_ratio(wp.GAMMA_V)
    p_up = 300e3
    below = orifice_mass_flow(p_up, 450.0, p_up * rc * (1 - 1e-9), **ARGS)
    above = orifice_mass_flow(p_up, 450.0, p_up * rc * (1 + 1e-9), **ARGS)
    assert below == pytest.approx(above, rel=1e-6)


def test_small_pressure_drop_is_incompressible():
    p_up, T, dp = 160e3, 420.0, 50.0
    m = orifice_mass_flow(p_up, T, p_up - dp, **ARGS)
    rho = p_up / (wp.R_V * T)
    assert m == pytest.approx(ARGS["cd"] * ARGS["area"] * np.sqrt(2 * rho * dp), rel=1e-3)
