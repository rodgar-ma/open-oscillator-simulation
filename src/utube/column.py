from dataclasses import dataclass
import numpy as np

G = 9.80665

@dataclass(frozen=True)
class Column:
  length: float = 10.0          # L [m]
  diameter: float = 0.5         # D [m]
  friction_factor: float = 0.02 # f [-]
  minor_loss: float = 1.5       # K [-]

  @property
  def area(self):
    return 0.25 * np.pi * self.diameter**2 # Area = pi*(diameter/2)^2

  @property
  def loss_coefficient(self):
    return self.friction_factor * self.length / (2 * self.diameter) + 0.5 * self.minor_loss

  @property
  def omega_0(self):
      return np.sqrt(2 * G / self.length)