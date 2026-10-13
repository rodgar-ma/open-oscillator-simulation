"""Open oscillator: high pipe + side tank with a gas cushion.

This follows the architecture Hydram Dynamics describes publicly
(https://hydram.is/technical): a liquid slug oscillates between a tall
"high pipe", whose top holds the steam, and a wider side tank whose top holds
a gas bubble. The oscillation is sustained by "alternately increasing the
pressure behind the liquid piston (in its direction of travel)". Geometry and
parameters below are illustrative, not Hydram's.

Coordinates (all heights from the bottom of the system)
-------------------------------------------------------
z     liquid level in the high pipe (cross-section A_h)
z_t   liquid level in the side tank (cross-section A_t);
      volume conservation gives z_t = z_t0 - r (z - z_ref), r = A_h / A_t
V_s   steam volume  = A_h (H_top - z) + V_dead
V_g   gas volume    = V_g0 + A_h (z - z_ref)

Dynamics (energy-based / Lagrangian)
------------------------------------
Kinetic energy T = 1/2 M(z) u^2 with the variable inertia
M(z) = rho A_h (z + L_c + r z_t(z)), so that

    M u' = A_h (p_g + p_assist - p_s) - rho g A_h (z - z_t) - 1/2 M'(z) u^2 - A_h dp_loss

The 1/2 M' u^2 term is what makes energy conserve while liquid moves between
sections of different area; the energy-closure tests would fail without it.
Losses are direction dependent: entering the high pipe from the tank costs an
entrance loss, leaving it into the tank costs a (Borda-Carnot) exit loss.

Thermodynamics of the steam space
---------------------------------
Same building blocks as ``model.py`` (uniform ideal-gas vapour, HKS interface
fluxes, flash boiling, fog, compressible check valve), plus a second interface:
the exposed pipe wall. The wall surface is wet and has its own temperature
node with a large (steel) heat capacity, held towards T_bulk by the waste-heat
exchangers at the top. Phase change on the wall uses an effective coefficient
h_cond: q = h_cond (T_sat(p_s) - T_w). Condensation on the walls is the loss
Hydram says forces them to build large diameters, and h_cond is the most
uncertain parameter of the model (clean film condensation on bare steel is of
order 1e4 W/m^2K; non-condensable gas or a low-conductivity liner lowers it
by orders of magnitude).

State: y = [z, u, m_v, T_v, T_s, T_w]

Energy bookkeeping and per-cycle metrics are in ``analysis.py``; the
reference case is in ``cases.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp

from .. import properties as wp
from ..valve import orifice_mass_flow

G = 9.81


@dataclass(frozen=True)
class Params:
    # geometry
    D_h: float = 0.6            # high pipe diameter, m
    H_top: float = 9.0          # top of the high pipe (steam space ceiling), m
    V_dead: float = 0.02        # extra steam volume at the top (valve, fittings), m^3
    L_c: float = 2.5            # length of the bottom connection (area A_h), m
    D_t: float = 1.4            # side tank diameter, m
    z_ref: float = 6.5          # reference high-pipe level (static equilibrium), m
    z_t0: float = 0.5           # tank level when z = z_ref, m
    V_g0: float = 2.0           # gas cushion volume when z = z_ref, m^3
    gamma_g: float = 1.4        # gas cushion: adiabatic ideal gas
    # liquid and waste heat
    rho_l: float = 983.0
    T_bulk: float = 333.15      # K
    # hydraulic losses: dp = 1/2 rho K v|v|
    K_sym: float = 0.7          # friction and bends, both directions
    K_in: float = 0.3           # tank -> high pipe (u > 0)
    K_out: float = 0.7          # high pipe -> tank (u < 0), Borda-Carnot ~ (1 - r)^2
    # free surface
    sigma: float = 0.04
    area_factor: float = 1.0
    delta: float = 1.0e-3       # surface layer thickness, m
    h_bulk: float = 2000.0      # W/(m^2 K)
    cav_area_factor: float = 10.0
    fog_tau: float | None = 0.01
    # wall (exposed part of the high pipe above the liquid)
    wall_wet: bool = True       # wall interface on/off
    h_cond: float = 300.0       # effective vapour <-> wet wall phase-change coefficient, W/(m^2 K)
    C_wall: float = 15.0e3      # areal heat capacity of the wall surface node, J/(m^2 K)
    h_wall_core: float = 500.0  # wall surface -> heated core (waste heat), W/(m^2 K)
    z_wall_min: float = 1.5     # lowest level that exposes wall (sizes the node), m
    # valve
    valve_area: float = 3.0e-3
    valve_cd: float = 0.8
    p_out: float = 125.0e3
    # drive: assist pressure in the direction of travel
    P_assist: float = 20.0e3    # Pa
    u_ref: float = 0.2          # m/s, smoothing of the sign function

    @property
    def A_h(self):
        return np.pi * self.D_h**2 / 4

    @property
    def A_t(self):
        return np.pi * self.D_t**2 / 4

    @property
    def r(self):
        return self.A_h / self.A_t

    @property
    def p_g0(self):
        """Gas pressure that balances saturated steam with the slug at z_ref."""
        return float(wp.p_sat(self.T_bulk)) + self.rho_l * G * (self.z_ref - self.z_t0)

    @property
    def A_wall_max(self):
        return np.pi * self.D_h * (self.H_top - self.z_wall_min) + self.A_h

    def with_(self, **kw):
        return replace(self, **kw)



@dataclass
class Fluxes:
    z: float
    u: float
    z_t: float
    M: float
    p_s: float          # steam pressure, Pa
    p_g: float          # gas cushion pressure, Pa
    p_assist: float     # drive pressure, Pa
    V_s: float
    V_g: float
    A_w: float          # exposed wall area, m^2
    mdot_surf: float    # free surface, + evaporation
    mdot_wall: float    # wall, + evaporation
    mdot_flash: float
    mdot_fog: float
    mdot_out: float
    Q_bulk: float       # bulk -> surface layer, W
    Q_core: float       # heated wall core -> wall node, W
    P_loss: float       # hydraulic dissipation, W


class Model:
    def __init__(self, p: Params):
        self.p = p

    def hks(self, T_l, p_v, T_v):
        s = self.p.sigma
        if s <= 0:
            return 0.0
        c = 2 * s / (2 - s) / np.sqrt(2 * np.pi * wp.R_V)
        return c * (wp.p_sat(T_l) / np.sqrt(T_l) - p_v / np.sqrt(T_v))

    def loss_dp(self, u):
        p = self.p
        K = p.K_sym + (p.K_in if u > 0 else p.K_out)
        return 0.5 * p.rho_l * K * u * abs(u)

    def fluxes(self, t, y) -> Fluxes:
        p = self.p
        z, u, m_v, T_v, T_s, T_w = y
        A_h = p.A_h
        z_t = p.z_t0 - p.r * (z - p.z_ref)
        M = p.rho_l * A_h * (z + p.L_c + p.r * z_t)
        V_s = A_h * (p.H_top - z) + p.V_dead
        V_g = p.V_g0 + A_h * (z - p.z_ref)
        if V_s <= 0 or V_g <= 0:
            raise RuntimeError(f"volume collapsed at t={t:.3f} s (z={z:.3f} m)")
        p_s = m_v * wp.R_V * T_v / V_s
        p_g = p.p_g0 * (p.V_g0 / V_g) ** p.gamma_g
        p_a = p.P_assist * np.tanh(u / p.u_ref)
        A_w = np.pi * p.D_h * max(p.H_top - z, 0.0) + A_h

        m_surf = p.area_factor * A_h * self.hks(T_s, p_s, T_v)
        # wet wall: condenses when T_w < T_sat(p_s), evaporates when T_w > T_sat(p_s)
        m_wall = (A_w * p.h_cond * (T_w - float(wp.T_sat(p_s))) / float(wp.h_fg(T_w))
                  if p.wall_wet else 0.0)
        m_flash = (p.cav_area_factor * A_h * max(self.hks(p.T_bulk, p_s, T_v), 0.0)
                   if p.cav_area_factor > 0 else 0.0)
        m_fog = (m_v / p.fog_tau * max(1.0 - float(wp.p_sat(T_v)) / p_s, 0.0)
                 if p.fog_tau else 0.0)
        m_out = orifice_mass_flow(p_s, T_v, p.p_out, p.valve_area, p.valve_cd, wp.R_V, wp.GAMMA_V)
        Q_bulk = p.h_bulk * p.area_factor * A_h * (p.T_bulk - T_s)
        Q_core = p.h_wall_core * p.A_wall_max * (p.T_bulk - T_w) if p.wall_wet else 0.0
        P_loss = abs(A_h * self.loss_dp(u) * u)
        return Fluxes(z, u, z_t, M, p_s, p_g, p_a, V_s, V_g, A_w, m_surf, m_wall,
                          m_flash, m_fog, m_out, Q_bulk, Q_core, P_loss)

    @staticmethod
    def _layer_flux(mdot, T_layer, T_v, T_bulk):
        """Energy into an interface layer from its phase change (see model.py)."""
        if mdot >= 0:
            return mdot * (wp.h_liquid(T_bulk) - wp.h_vapour(T_layer))
        return -mdot * (wp.h_vapour(T_v) - wp.h_liquid(T_layer))

    def rhs(self, t, y):
        p = self.p
        f = self.fluxes(t, y)
        z, u, m_v, T_v, T_s, T_w = y
        A_h = p.A_h

        # slug dynamics
        dM = p.rho_l * A_h * (1 - p.r**2)
        force = (A_h * (f.p_g + f.p_assist - f.p_s) - p.rho_l * G * A_h * (z - f.z_t)
                 - 0.5 * dM * u**2 - A_h * self.loss_dp(u))
        du = force / f.M

        # steam space
        H_in = 0.0
        for mdot, T_l in ((f.mdot_surf, T_s), (f.mdot_wall, T_w)):
            H_in += mdot * (wp.h_vapour(T_l) if mdot >= 0 else wp.h_vapour(T_v))
        H_in += f.mdot_flash * wp.h_vapour(p.T_bulk)
        H_out = f.mdot_out * wp.h_vapour(T_v) + f.mdot_fog * wp.h_liquid(T_v)
        dU = H_in - H_out + f.p_s * A_h * u
        dm = f.mdot_surf + f.mdot_wall + f.mdot_flash - f.mdot_fog - f.mdot_out
        dT_v = (dU - wp.u_vapour(T_v) * dm) / (m_v * wp.CV_V)

        # interface nodes
        C_s = p.rho_l * p.area_factor * A_h * p.delta * wp.C_L
        dT_s = (f.Q_bulk + self._layer_flux(f.mdot_surf, T_s, T_v, p.T_bulk)) / C_s
        if p.wall_wet:
            C_w = p.C_wall * p.A_wall_max
            dT_w = (f.Q_core + self._layer_flux(f.mdot_wall, T_w, T_v, p.T_bulk)) / C_w
        else:
            dT_w = 0.0
        return [u, du, dm, dT_v, dT_s, dT_w]

    def initial_state(self, z0=8.0, u0=0.0):
        p = self.p
        T = p.T_bulk
        V_s = p.A_h * (p.H_top - z0) + p.V_dead
        m_v = float(wp.p_sat(T)) * V_s / (wp.R_V * T)
        return np.array([z0, u0, m_v, T, T, T])


@dataclass
class Result:
    model: Model
    t: np.ndarray
    y: np.ndarray
    sol: object = field(repr=False)

    def flux_series(self, t=None):
        if t is None:
            t, y = self.t, self.y
        else:
            y = self.sol.sol(t)
        rows = [self.model.fluxes(ti, y[:, i]) for i, ti in enumerate(t)]
        out = {k: np.array([getattr(r, k) for r in rows]) for k in Fluxes.__dataclass_fields__}
        out["t"] = np.asarray(t)
        out["m_v"], out["T_v"], out["T_s"], out["T_w"] = y[2], y[3], y[4], y[5]
        return out

    def top_times(self, t_min=0.0):
        """Times of the top dead centre (u crosses 0 downwards), via dense output."""
        t = np.linspace(max(t_min, self.t[0]), self.t[-1], 40 * len(self.t))
        u = self.sol.sol(t)[1]
        i = np.where((u[:-1] > 0) & (u[1:] <= 0))[0]
        return t[i] - u[i] * (t[i + 1] - t[i]) / (u[i + 1] - u[i])


def simulate(p: Params, t_end: float, z0=8.0, rtol=1e-8, max_step=0.01,
                  n_out=6000) -> Result:
    m = Model(p)
    y0 = m.initial_state(z0)
    atol = 1e-9 * (np.abs(y0) + 1e-3)
    sol = solve_ivp(m.rhs, (0, t_end), y0, method="Radau", rtol=rtol, atol=atol,
                    max_step=max_step, dense_output=True, t_eval=np.linspace(0, t_end, n_out))
    if not sol.success:
        raise RuntimeError(sol.message)
    return Result(m, sol.t, sol.y, sol)
