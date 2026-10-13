"""Open oscillator (high pipe + side tank with gas cushion).

1. Conservative limit: no phase change, no losses, no drive, no valve.
   Total energy (variable-inertia slug + two gas springs + gravity) must be
   conserved; this checks the 1/2 M'(z) u^2 term of the Lagrangian.
2. Small-amplitude frequency against the linearised stiffness.
3. Reference case: energy closure, second law, periodicity, valve logic.
4. Documented finding: with strongly wetted walls a small pipe never ejects.
"""

import numpy as np
import pytest

from liquid_piston import properties as wp
from liquid_piston.open_oscillator import (G, REFERENCE, cycle_metrics, energy_error,
                                           scaled_reference, simulate)

CONSERVATIVE = REFERENCE.with_(sigma=0.0, cav_area_factor=0.0, fog_tau=None, wall_wet=False,
                                    K_sym=0.0, K_in=0.0, K_out=0.0, P_assist=0.0, valve_area=0.0)


def test_conservative_energy():
    res = simulate(CONSERVATIVE, 20.0, z0=8.3)
    assert energy_error(res) < 1e-7
    assert np.ptp(res.y[0]) > 1.5  # large, nonlinear oscillation with varying inertia


def test_small_amplitude_frequency():
    p = CONSERVATIVE
    p_s0 = float(wp.p_sat(p.T_bulk))
    V_s0 = p.A_h * (p.H_top - p.z_ref) + p.V_dead
    k = (p.rho_l * G * p.A_h * (1 + p.r)
         + p.gamma_g * p.p_g0 * p.A_h**2 / p.V_g0
         + wp.GAMMA_V * p_s0 * p.A_h**2 / V_s0)
    M = p.rho_l * p.A_h * (p.z_ref + p.L_c + p.r * p.z_t0)
    T_th = 2 * np.pi / np.sqrt(k / M)
    res = simulate(p, 8 * T_th, z0=p.z_ref + 1e-3, n_out=20000)
    z = res.y[0] - 0.5 * (res.y[0].max() + res.y[0].min())
    i = np.where(np.diff(np.sign(z)) != 0)[0]
    zc = res.t[i] - z[i] * (res.t[i + 1] - res.t[i]) / (z[i + 1] - z[i])
    assert 2 * np.mean(np.diff(zc)) == pytest.approx(T_th, rel=1e-3)


@pytest.fixture(scope="module")
def ref():
    res = simulate(REFERENCE, 60.0)
    return res, cycle_metrics(res), cycle_metrics(res, which=-2)


def test_reference_ejects_with_cop_above_one(ref):
    _, m, _ = ref
    assert m.m_out > 0 and m.cop > 1.0
    assert m.p_s_max > REFERENCE.p_out


def test_reference_energy_and_second_law(ref):
    res, m, _ = ref
    assert energy_error(res) < 1e-5
    assert abs(m.closure) < 1e-4
    assert 0.0 < m.second_law_efficiency < 1.0


def test_reference_is_a_limit_cycle(ref):
    _, m, prev = ref
    assert m.m_out == pytest.approx(prev.m_out, rel=1e-2)
    assert m.period == pytest.approx(prev.period, rel=1e-3)


def test_valve_only_opens_above_outlet_pressure(ref):
    res, _, _ = ref
    fx = res.flux_series()
    assert np.all(fx["p_s"][fx["mdot_out"] > 0] > REFERENCE.p_out)


def test_small_wet_pipe_does_not_eject():
    p = scaled_reference(0.3, h_cond=2000.0)
    m = cycle_metrics(simulate(p, 60.0))
    assert m.m_out == 0.0
