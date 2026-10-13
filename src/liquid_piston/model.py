"""Zero-dimensional model of a liquid piston oscillating in a U-tube.

Physical picture
----------------
A liquid column of length ``L`` fills a U-tube of cross-section ``A``. Leg 1
ends in a closed vapour chamber; leg 2 is driven by a prescribed pressure (or
the column motion is prescribed directly). ``x`` is the rise of the liquid
level in leg 1, so the vapour volume is ``V = V0 - A x``.

* When the column falls, the chamber expands, the vapour pressure drops below
  saturation and liquid flashes at the free surface.
* When it rises, the vapour is compressed and heated; part of it condenses
  back on the surface, and once the chamber pressure exceeds the outlet
  pressure the check valve opens and steam is ejected.

State vector
------------
dynamic mode:    y = [x, v, m_v, T_v, T_s]
kinematic mode:  y = [m_v, T_v, T_s]          (x, v, a prescribed)

``m_v`` and ``T_v`` are the vapour mass and temperature (ideal gas, uniform),
``T_s`` the temperature of a thin liquid surface layer that exchanges heat
with the bulk liquid, held at ``T_bulk`` by the waste-heat supply.

Every sub-model can be switched off, which is how the verification ladder in
``tests/`` checks each rung against an analytical result:

* ``sigma = 0``           no phase change at the free surface
* ``cav_area_factor = 0`` no bulk flash boiling (cavitation)
* ``fog_tau = None``      no homogeneous condensation in the vapour
* ``valve_area = 0``      no ejection
* ``h_wall = 0``          adiabatic chamber walls
* ``chamber = "open"``    constant pressure above leg 1 (plain U-tube)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from . import properties as wp
from .valve import orifice_mass_flow

G = 9.81  # m/s^2


# ---------------------------------------------------------------------------
# Parameters and drives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Model parameters (SI units). Defaults describe an illustrative
    multi-tonne device; they are not Hydram's design values."""

    # geometry
    L: float = 10.0            # liquid column length, m
    D: float = 1.0             # pipe diameter, m
    H0: float = 2.0            # vapour headspace height at x = 0, m
    # liquid
    rho_l: float = 983.0       # kg/m^3 (water at ~60 degC)
    T_bulk: float = 333.15     # bulk liquid temperature held by waste heat, K
    # friction: pressure loss = K * rho * v|v| / 2  (K includes f L/D)
    K_loss: float = 2.0
    # phase change at the free surface (Hertz-Knudsen-Schrage)
    sigma: float = 0.04        # accommodation coefficient, 0..1
    area_factor: float = 1.0   # free-surface area / pipe cross-section
    # bulk flash boiling (cavitation) while the chamber is below p_sat(T_bulk):
    # one-way HKS evaporation from bubble interfaces, area = cav_area_factor * A
    cav_area_factor: float = 0.0
    # homogeneous condensation (fog) of supersaturated vapour: relaxation
    # time towards p_v = p_sat(T_v); droplets rain out to the pool. None = off
    fog_tau: float | None = None
    # surface layer
    delta: float = 1.0e-3      # layer thickness, m
    h_bulk: float = 2000.0     # layer-to-bulk heat transfer coefficient, W/(m^2 K)
    # chamber walls (sensible heat only, walls at T_bulk)
    h_wall: float = 0.0        # W/(m^2 K)
    A_wall: float | None = None  # m^2; default: side wall of headspace + lid
    # ejection valve
    valve_area: float = 2.0e-3  # m^2
    valve_cd: float = 0.8
    p_out: float = 155.0e3     # outlet (steam header) pressure, Pa
    # chamber type
    chamber: Literal["closed", "open"] = "closed"
    p_open: float = 101.325e3  # pressure above leg 1 when chamber == "open", Pa

    # --- derived quantities ---
    @property
    def A(self) -> float:
        return np.pi * self.D**2 / 4.0

    @property
    def V0(self) -> float:
        return self.A * self.H0

    @property
    def m_liquid(self) -> float:
        return self.rho_l * self.A * self.L

    @property
    def A_interface(self) -> float:
        return self.area_factor * self.A

    @property
    def A_cavitation(self) -> float:
        return self.cav_area_factor * self.A

    @property
    def m_layer(self) -> float:
        return self.rho_l * self.A_interface * self.delta

    @property
    def wall_area(self) -> float:
        if self.A_wall is not None:
            return self.A_wall
        return np.pi * self.D * self.H0 + self.A

    @property
    def omega_hydrostatic(self) -> float:
        """Natural angular frequency of the plain U-tube, rad/s."""
        return np.sqrt(2.0 * G / self.L)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


@dataclass(frozen=True)
class PressureDrive:
    """Prescribed pressure on leg 2: p2(t) = p_mean + p_amp sin(2 pi f t)."""

    p_mean: float
    p_amp: float = 0.0
    freq: float = 0.0

    def p2(self, t):
        return self.p_mean + self.p_amp * np.sin(2.0 * np.pi * self.freq * t)


@dataclass(frozen=True)
class KinematicDrive:
    """Prescribed motion: x(t) = x_center - stroke * cos(2 pi f t).

    The column starts at its lowest point (largest vapour volume). The drive
    pressure needed to produce this motion is computed, not imposed.
    """

    x_center: float
    stroke: float
    freq: float

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq

    @property
    def period(self) -> float:
        return 1.0 / self.freq

    def kinematics(self, t):
        w = self.omega
        c, s = np.cos(w * t), np.sin(w * t)
        return (self.x_center - self.stroke * c,
                self.stroke * w * s,
                self.stroke * w**2 * c)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class Fluxes:
    """Instantaneous rates (W or kg/s) at one instant. Signs: + into the
    named destination."""

    x: float
    v: float
    a: float
    p_v: float          # chamber pressure, Pa
    p2: float           # drive pressure on leg 2, Pa
    V: float            # vapour volume, m^3
    mdot_evap: float    # free surface: + evaporation, - condensation, kg/s
    mdot_flash: float   # bulk flash boiling (>= 0), kg/s
    mdot_fog: float     # homogeneous condensation in the vapour (>= 0), kg/s
    mdot_out: float     # ejected steam, kg/s
    Q_bulk: float       # bulk -> surface layer, W
    Q_wall: float       # walls -> vapour, W
    P_fric: float       # friction dissipation (>= 0), W


class Model:
    def __init__(self, params: Params, drive: PressureDrive | KinematicDrive):
        self.p = params
        self.drive = drive
        self.kinematic = isinstance(drive, KinematicDrive)

    # --- sub-models --------------------------------------------------------
    def interface_mass_flux(self, T_s, p_v, T_v):
        """Hertz-Knudsen-Schrage net evaporation flux, kg/(m^2 s)."""
        s = self.p.sigma
        if s <= 0.0:
            return 0.0
        coeff = 2.0 * s / (2.0 - s) / np.sqrt(2.0 * np.pi * wp.R_V)
        return coeff * (wp.p_sat(T_s) / np.sqrt(T_s) - p_v / np.sqrt(T_v))

    def friction_pressure(self, v):
        return 0.5 * self.p.K_loss * self.p.rho_l * v * abs(v)

    # --- unpacking ---------------------------------------------------------
    def _unpack(self, t, y):
        if self.kinematic:
            x, v, a = self.drive.kinematics(t)
            m_v, T_v, T_s = y
        else:
            x, v, m_v, T_v, T_s = y
            a = np.nan
        return x, v, a, m_v, T_v, T_s

    def fluxes(self, t, y) -> Fluxes:
        """All rates at time ``t``. Used by the RHS and by post-processing."""
        p = self.p
        x, v, a, m_v, T_v, T_s = self._unpack(t, y)
        A = p.A
        P_fric = abs(A * self.friction_pressure(v) * v)

        if p.chamber == "open":
            p_v, V = p.p_open, np.nan
            mdot_evap = mdot_flash = mdot_fog = mdot_out = Q_bulk = Q_wall = 0.0
        else:
            V = p.V0 - A * x
            if V <= 0.0:
                raise RuntimeError(f"vapour volume collapsed at t={t:.4f} s (x={x:.3f} m)")
            p_v = m_v * wp.R_V * T_v / V
            mdot_evap = p.A_interface * self.interface_mass_flux(T_s, p_v, T_v)
            mdot_flash = (p.A_cavitation * max(self.interface_mass_flux(p.T_bulk, p_v, T_v), 0.0)
                          if p.cav_area_factor > 0.0 else 0.0)
            mdot_fog = (m_v / p.fog_tau * max(1.0 - float(wp.p_sat(T_v)) / p_v, 0.0)
                        if p.fog_tau else 0.0)
            mdot_out = orifice_mass_flow(p_v, T_v, p.p_out, p.valve_area, p.valve_cd,
                                         wp.R_V, wp.GAMMA_V)
            Q_bulk = p.h_bulk * p.A_interface * (p.T_bulk - T_s)
            Q_wall = p.h_wall * p.wall_area * (p.T_bulk - T_v)

        if self.kinematic:
            # pressure on leg 2 required to impose the prescribed motion
            p2 = p_v + (p.m_liquid * a + 2.0 * p.rho_l * G * A * x) / A + self.friction_pressure(v)
        else:
            p2 = self.drive.p2(t)

        return Fluxes(x, v, a, p_v, p2, V, mdot_evap, mdot_flash, mdot_fog, mdot_out, Q_bulk, Q_wall, P_fric)

    # --- right-hand side -----------------------------------------------------
    def rhs(self, t, y):
        p = self.p
        f = self.fluxes(t, y)
        _, _, _, m_v, T_v, T_s = self._unpack(t, y)

        if p.chamber == "open":
            dm = dT_v = dT_s = 0.0
        else:
            # vapour: open-system energy balance on the chamber
            me = f.mdot_evap
            H_in = me * (wp.h_vapour(T_s) if me >= 0.0 else wp.h_vapour(T_v))
            H_in += f.mdot_flash * wp.h_vapour(p.T_bulk)
            # fog: droplets leave as liquid at T_v, so the latent heat stays in the vapour
            H_out = f.mdot_out * wp.h_vapour(T_v) + f.mdot_fog * wp.h_liquid(T_v)
            dV = -p.A * f.v
            dU = H_in - H_out - f.p_v * dV + f.Q_wall
            dm = me + f.mdot_flash - f.mdot_fog - f.mdot_out
            dT_v = (dU - wp.u_vapour(T_v) * dm) / (m_v * wp.CV_V)
            # surface layer: fixed mass, replenished from / returned to bulk
            if me >= 0.0:   # evaporation: bulk liquid in, vapour at T_s out
                E_flux = me * (wp.h_liquid(p.T_bulk) - wp.h_vapour(T_s))
            else:           # condensation: vapour at T_v in, liquid at T_s to bulk
                E_flux = -me * (wp.h_vapour(T_v) - wp.h_liquid(T_s))
            dT_s = (f.Q_bulk + E_flux) / (p.m_layer * wp.C_L)

        if self.kinematic:
            return [dm, dT_v, dT_s]

        force = p.A * (f.p2 - f.p_v) - 2.0 * p.rho_l * G * p.A * f.x \
            - p.A * self.friction_pressure(f.v)
        return [f.v, force / p.m_liquid, dm, dT_v, dT_s]

    # --- initial state ---------------------------------------------------------
    def initial_state(self, x0=0.0, v0=0.0, T_v0=None, T_s0=None, p_v0=None):
        """Saturated vapour at the bulk temperature unless overridden."""
        p = self.p
        T_v0 = p.T_bulk if T_v0 is None else T_v0
        T_s0 = p.T_bulk if T_s0 is None else T_s0
        p_v0 = float(wp.p_sat(T_s0)) if p_v0 is None else p_v0
        if self.kinematic:
            x0, _, _ = self.drive.kinematics(0.0)
        V = p.V0 - p.A * x0 if p.chamber == "closed" else 1.0
        m_v0 = p_v0 * V / (wp.R_V * T_v0)
        if self.kinematic:
            return np.array([m_v0, T_v0, T_s0])
        return np.array([x0, v0, m_v0, T_v0, T_s0])


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
@dataclass
class Result:
    model: Model
    t: np.ndarray
    y: np.ndarray
    sol: object = field(repr=False)

    def flux_series(self, t=None):
        """Evaluate every flux on a time grid (default: solver output grid)."""
        if t is None:
            t, y = self.t, self.y
        else:
            y = self.sol.sol(t)
        rows = [self.model.fluxes(ti, y[:, i]) for i, ti in enumerate(t)]
        out = {k: np.array([getattr(r, k) for r in rows]) for k in Fluxes.__dataclass_fields__}
        out["t"] = np.asarray(t)
        m = self.model
        if m.kinematic:
            out["m_v"], out["T_v"], out["T_s"] = y
        else:
            out["m_v"], out["T_v"], out["T_s"] = y[2], y[3], y[4]
        return out


def simulate(params: Params, drive, t_end: float, y0=None, method="Radau",
             rtol=1e-8, atol=None, max_step=None, n_out=4000, **init_kw) -> Result:
    model = Model(params, drive)
    if y0 is None:
        y0 = model.initial_state(**init_kw)
    if atol is None:
        scale = np.abs(y0) + 1e-3
        atol = 1e-9 * scale
    if max_step is None:
        period = (1.0 / drive.freq if getattr(drive, "freq", 0) else
                  2.0 * np.pi / params.omega_hydrostatic)
        max_step = period / 200.0
    sol = solve_ivp(model.rhs, (0.0, t_end), y0, method=method, rtol=rtol, atol=atol,
                    max_step=max_step, dense_output=True,
                    t_eval=np.linspace(0.0, t_end, n_out))
    if not sol.success:
        raise RuntimeError(sol.message)
    return Result(model, sol.t, sol.y, sol)
