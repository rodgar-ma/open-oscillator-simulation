"""Reference case used by the scripts, the tests and the technical note.

An illustrative multi-tonne device (D = 1 m, L = 10 m, ~7.7 t of water at
60 degC) driven at the natural frequency of the plain U-tube. These are
plausible orders of magnitude chosen for the study, not Hydram's design.
"""

from __future__ import annotations

import numpy as np

from .model import KinematicDrive, Params

REFERENCE = Params(
    L=10.0, D=1.0, H0=2.0,
    T_bulk=273.15 + 60.0,
    K_loss=2.0,
    sigma=0.04, area_factor=1.0,
    cav_area_factor=10.0,
    fog_tau=0.01,
    delta=1.0e-3, h_bulk=2000.0,
    h_wall=0.0,
    valve_area=2.0e-3, valve_cd=0.8, p_out=155.0e3,
)


def reference_drive(params: Params = REFERENCE, stroke: float = 1.3,
                    x_center: float = 0.5, freq_ratio: float = 1.0) -> KinematicDrive:
    """Stroke between x = -0.8 m and x = 1.8 m: headspace 2.8 m -> 0.2 m."""
    f0 = params.omega_hydrostatic / (2.0 * np.pi)
    return KinematicDrive(x_center=x_center, stroke=stroke, freq=freq_ratio * f0)
