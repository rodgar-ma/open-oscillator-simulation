import numpy as np
from utube.column import Column
from utube.dynamics import simulate, specific_energy

COL = Column()
X0 = 0.5
T_END = 10 * 2 * np.pi / COL.omega_0

def test_matches_analytic_solution():
    sol = simulate(COL, X0, T_END)
    exact = X0 * np.cos(COL.omega_0 * sol.t)
    assert np.max(np.abs(sol.y[0] - exact)) < 1e-8

def test_energy_is_conserved():
    sol = simulate(COL, X0, T_END)
    e = specific_energy(sol.y[0], sol.y[1], COL)
    assert (e.max() - e.min()) / e[0] < 1e-8

def test_frequency_is_independent_of_amplitude():
    periods = []
    for x0 in (0.1, 0.5, 1.0):
        sol = simulate(COL, x0, T_END)
        crossings = np.where(np.diff(np.signbit(sol.y[0])))[0]
        periods.append(2 * np.mean(np.diff(sol.t[crossings])))
    assert np.allclose(periods, periods[0], rtol=1e-4)

def test_quadratic_damping_envelope():
  sol = simulate(COL, X0, T_END, friction=True)
  k = 4 * COL.loss_coefficient * COL.omega_0 / (3 * np.pi * COL.length)
  predicted = X0 / (1 + k * X0 * sol.t[-1])
  observed = np.max(np.abs(sol.y[0][-400:]))
  assert abs(observed - predicted) / predicted < 0.02