"""Reference case of the open oscillator and geometric scaling.

A 0.6 m x 9 m high pipe with a 1.4 m side tank, water at 60 degC and an
outlet at 125 kPa: orders of magnitude inspired by Hydram's public
description, not their design values.
"""

from __future__ import annotations

from .model import Params

REFERENCE = Params()


def scaled_reference(D_h: float, base: Params = REFERENCE, **overrides) -> Params:
    """Geometrically similar unit with high-pipe diameter ``D_h`` (same height).

    Tank diameter scales with D_h; gas volume, valve area and dead volume
    scale with the pipe cross-section.
    """
    k = D_h / base.D_h
    return base.with_(D_h=D_h, D_t=base.D_t * k, V_g0=base.V_g0 * k**2,
                      valve_area=base.valve_area * k**2, V_dead=base.V_dead * k**2,
                      **overrides)
