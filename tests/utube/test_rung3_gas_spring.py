"""Rung 3: closed chamber, no phase change, adiabatic walls, no valve.

The vapour behaves as an ideal-gas spring:
  * p V^gamma is constant (isentropic),
  * small oscillations have omega^2 = (2 rho g A + gamma p0 A^2 / V0) / m,
  * total energy (mechanical + vapour internal energy) is conserved.
"""

import numpy as np
import pytest

from liquid_piston.utube import Params, PressureDrive, balances, simulate
from liquid_piston import properties as wp


def gas_spring():
    return Params(sigma=0.0, valve_area=0.0, h_wall=0.0, K_loss=0.0)


def run(x0, n_periods=8):
    p = gas_spring()
    p0 = float(wp.p_sat(p.T_bulk))
    drive = PressureDrive(p0)  # leg 2 balances the chamber at x = 0
    k = 2 * p.rho_l * 9.81 * p.A + wp.GAMMA_V * p0 * p.A**2 / p.V0
    w_th = np.sqrt(k / p.m_liquid)
    res = simulate(p, drive, n_periods * 2 * np.pi / w_th, x0=x0, n_out=20000)
    return p, res, w_th


def test_isentropic_compression():
    p, res, _ = run(x0=0.8)
    fx = res.flux_series()
    pv_gamma = fx["p_v"] * fx["V"] ** wp.GAMMA_V
    assert np.ptp(pv_gamma) / pv_gamma[0] < 1e-6
    assert fx["p_v"].max() / fx["p_v"].min() > 2  # large-amplitude, non-trivial


def test_small_amplitude_frequency():
    p, res, w_th = run(x0=1e-3)
    t, x = res.t, res.y[0]
    x = x - 0.5 * (x.max() + x.min())  # oscillation is about the gas-spring equilibrium
    i = np.where(np.diff(np.sign(x)) != 0)[0]
    zc = t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
    assert 2 * np.mean(np.diff(zc)) == pytest.approx(2 * np.pi / w_th, rel=1e-4)


def test_energy_conserved_with_gas_spring():
    _, res, _ = run(x0=0.8)
    assert balances(res).energy_error < 1e-6
