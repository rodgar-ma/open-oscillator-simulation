"""Rung 0: property model against IAPWS-IF97 verification values and CoolProp."""

import numpy as np
import pytest

from liquid_piston import properties as wp

# Verification tables 35 and 36 of the IAPWS-IF97 release
IF97_PSAT = [(300.0, 0.353658941e-2), (500.0, 0.263889776e1), (600.0, 0.123443146e2)]
IF97_TSAT = [(0.1, 0.372755919e3), (1.0, 0.453035632e3), (10.0, 0.584149488e3)]


@pytest.mark.parametrize("T, p_mpa", IF97_PSAT)
def test_psat_if97_verification(T, p_mpa):
    assert wp.p_sat(T) / 1e6 == pytest.approx(p_mpa, rel=1e-8)


@pytest.mark.parametrize("p_mpa, T", IF97_TSAT)
def test_tsat_if97_verification(p_mpa, T):
    assert wp.T_sat(p_mpa * 1e6) == pytest.approx(T, rel=1e-8)


def test_psat_tsat_roundtrip():
    T = np.linspace(275.0, 640.0, 50)
    np.testing.assert_allclose(wp.T_sat(wp.p_sat(T)), T, rtol=1e-9)


def test_latent_heat_exact_at_reference():
    assert wp.h_fg(373.15) == pytest.approx(2256.4e3)


def test_against_coolprop():
    CP = pytest.importorskip("CoolProp.CoolProp")
    T = np.linspace(280.0, 400.0, 13)
    p_ref = np.array([CP.PropsSI("P", "T", t, "Q", 0, "Water") for t in T])
    hfg_ref = np.array([CP.PropsSI("H", "T", t, "Q", 1, "Water")
                        - CP.PropsSI("H", "T", t, "Q", 0, "Water") for t in T])
    np.testing.assert_allclose(wp.p_sat(T), p_ref, rtol=1e-3)
    # simple caloric model: latent heat within 1 % from 7 to 127 degC
    np.testing.assert_allclose(wp.h_fg(T), hfg_ref, rtol=1e-2)


def test_vapour_ideal_gas_density_vs_coolprop():
    """Ideal-gas vapour density is within 2 % of real steam at the operating
    points of interest (saturated or superheated, up to ~200 kPa)."""
    CP = pytest.importorskip("CoolProp.CoolProp")
    for p, T in [(20e3, 340.0), (155e3, 395.0), (155e3, 500.0), (200e3, 450.0)]:
        rho_ref = CP.PropsSI("D", "P", p, "T", T, "Water")
        assert wp.vapour_density(p, T) == pytest.approx(rho_ref, rel=2e-2)
