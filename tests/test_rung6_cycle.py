"""Rung 6: the full cycle (flash boiling, surface exchange, fog, ejection).

No analytical solution exists here, so the checks are physical and numerical:
  * first law over the periodic cycle and over the whole transient,
  * second law: exergy delivered to the water stream <= drive work,
  * consistency: the valve only opens above p_out, ejected steam is not wet,
  * numerical convergence: results do not depend on the solver tolerance.
"""

import numpy as np
import pytest

from liquid_piston import REFERENCE, balances, cycle_metrics, reference_drive, run_to_periodic
from liquid_piston import properties as wp

N_CYCLES = 12


@pytest.fixture(scope="module")
def ref():
    res = run_to_periodic(REFERENCE, reference_drive(), n_cycles=N_CYCLES)
    return res, cycle_metrics(res)


def test_device_produces_steam(ref):
    _, m = ref
    assert m.m_out > 0.0
    assert m.p_max > REFERENCE.p_out


def test_cycle_is_periodic_and_first_law_closes(ref):
    _, m = ref
    assert abs(m.closure) < 1e-4


def test_transient_energy_and_mass_balance(ref):
    res, _ = ref
    b = balances(res)
    assert b.energy_error < 1e-5
    assert b.mass_error < 1e-3


def test_second_law(ref):
    _, m = ref
    assert 0.0 < m.second_law_efficiency < 1.0
    assert m.W > m.exergy_out


def test_ejected_steam_is_superheated(ref):
    _, m = ref
    assert m.T_steam > wp.T_sat(REFERENCE.p_out)


def test_fog_keeps_vapour_near_saturation(ref):
    _, m = ref
    assert m.min_superheat > -0.5


def test_valve_only_opens_above_outlet_pressure(ref):
    res, _ = ref
    fx = res.flux_series()
    assert np.all(fx["p_v"][fx["mdot_out"] > 0] > REFERENCE.p_out)


def test_solver_tolerance_convergence(ref):
    _, m_fine = ref
    m_coarse = cycle_metrics(run_to_periodic(REFERENCE, reference_drive(),
                                             n_cycles=N_CYCLES, rtol=1e-6))
    assert m_coarse.m_out == pytest.approx(m_fine.m_out, rel=1e-3)
    assert m_coarse.W == pytest.approx(m_fine.W, rel=1e-3)


def test_no_steam_without_flash_boiling():
    """With only the free surface, evaporation is too slow to refill the
    chamber and the valve never opens (documented finding)."""
    m = cycle_metrics(run_to_periodic(REFERENCE.with_(cav_area_factor=0.0),
                                      reference_drive(), n_cycles=N_CYCLES))
    assert m.m_out == 0.0
