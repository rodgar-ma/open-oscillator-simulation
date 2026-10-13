"""Rungs 1-2: plain U-tube (constant pressure above both legs).

1. Frictionless: period 2 pi sqrt(L / 2g) and conserved mechanical energy.
2. Quadratic friction: amplitude decay against the first-order energy
   method, a(t) = a0 / (1 + (4 / 3 pi) c omega a0 t), with c = K / (2L).
"""

import numpy as np
import pytest

from liquid_piston import Params, PressureDrive, balances, simulate

P_ATM = 101.325e3


def plain_utube(**kw):
    return Params(chamber="open", p_open=P_ATM, **kw)


def zero_crossings(t, x):
    i = np.where(np.diff(np.sign(x)) != 0)[0]
    return t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])


def test_frictionless_period():
    p = plain_utube(K_loss=0.0)
    T_th = 2 * np.pi / p.omega_hydrostatic
    res = simulate(p, PressureDrive(P_ATM), 10 * T_th, x0=0.5, n_out=20000)
    zc = zero_crossings(res.t, res.y[0])
    assert 2 * np.mean(np.diff(zc)) == pytest.approx(T_th, rel=1e-5)


def test_frictionless_energy_conserved():
    p = plain_utube(K_loss=0.0)
    T_th = 2 * np.pi / p.omega_hydrostatic
    res = simulate(p, PressureDrive(P_ATM), 10 * T_th, x0=0.5)
    assert balances(res).energy_error < 1e-6


def test_quadratic_friction_decay():
    p = plain_utube(K_loss=3.0)
    w, a0 = p.omega_hydrostatic, 0.5
    c = p.K_loss / (2 * p.L)
    T_th = 2 * np.pi / w
    res = simulate(p, PressureDrive(P_ATM), 12 * T_th, x0=a0, n_out=40000)
    t, x = res.t, res.y[0]
    # successive maxima of x
    i = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    a_th = a0 / (1 + 4 / (3 * np.pi) * c * w * a0 * t[i])
    np.testing.assert_allclose(x[i], a_th, rtol=0.03)
    assert x[i][-1] < 0.5 * a0  # decay is significant, so the test has teeth


def test_friction_energy_balance_closes():
    p = plain_utube(K_loss=3.0)
    res = simulate(p, PressureDrive(P_ATM, p_amp=2e3, freq=0.2), 40.0, x0=0.3)
    assert balances(res).energy_error < 1e-5
