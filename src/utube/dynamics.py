import numpy as np
from scipy.integrate import solve_ivp
from .column import Column, G

DEFAULT_N = 200

def rhs(t, y, col, friction=False):
  x, v = y
  drag = col.loss_coefficient * v * abs(v) if friction else 0.0
  return [v, (-2.0 * G * x - drag) / col.length]

def specific_energy(x, v, col):
  return 0.5 * col.length * v**2 + G * x**2

def simulate(col, x0, t_end, friction=False, n=None):
  n = n or int(t_end * DEFAULT_N)
  return solve_ivp(
    rhs, (0.0, t_end), [x0, 0.0],
    args=(col, friction), method="RK45",
    t_eval=np.linspace(0.0, t_end, n),
    rtol=1e-10, atol=1e-12)