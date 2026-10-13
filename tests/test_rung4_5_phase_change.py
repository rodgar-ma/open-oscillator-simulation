"""Rungs 4-5: phase change at the free surface and the surface layer.

4. A still chamber with sub-saturated vapour relaxes to p_sat(T_bulk).
5. Mass and energy balances close with evaporation, condensation and the
   surface-layer heat exchange all active.
"""

import numpy as np
import pytest

from liquid_piston import KinematicDrive, Params, balances, simulate
from liquid_piston import properties as wp


def test_relaxes_to_saturation():
    p = Params(valve_area=0.0, h_wall=50.0)
    still = KinematicDrive(x_center=0.0, stroke=0.0, freq=0.2)
    p_sat = float(wp.p_sat(p.T_bulk))
    res = simulate(p, still, 600.0, p_v0=0.5 * p_sat)
    fx = res.flux_series()
    assert fx["p_v"][-1] == pytest.approx(p_sat, rel=1e-3)
    assert fx["T_s"][-1] == pytest.approx(p.T_bulk, abs=0.01)
    assert fx["T_v"][-1] == pytest.approx(p.T_bulk, abs=0.05)


def test_evaporative_cooling_of_surface():
    """Evaporation into a sub-saturated chamber must cool the surface layer."""
    p = Params(valve_area=0.0)
    still = KinematicDrive(x_center=0.0, stroke=0.0, freq=0.2)
    res = simulate(p, still, 2.0, p_v0=0.5 * float(wp.p_sat(p.T_bulk)))
    assert res.y[2].min() < p.T_bulk - 1.0


@pytest.mark.parametrize("cav, fog", [(0.0, None), (10.0, None), (10.0, 0.01)])
def test_balances_close_with_phase_change(cav, fog):
    p = Params(valve_area=0.0, cav_area_factor=cav, fog_tau=fog, h_wall=20.0)
    drive = KinematicDrive(x_center=0.5, stroke=1.3, freq=0.22)
    res = simulate(p, drive, 3 / drive.freq)
    b = balances(res)
    assert b.energy_error < 1e-5
    assert b.mass_error < 1e-3
