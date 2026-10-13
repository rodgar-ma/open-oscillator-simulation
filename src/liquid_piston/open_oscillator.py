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
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from . import properties as wp
from .valve import orifice_mass_flow

G = 9.81


@dataclass(frozen=True)
class OpenParams:
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


OPEN_REFERENCE = OpenParams()


@dataclass
class OpenFluxes:
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


class OpenModel:
    def __init__(self, p: OpenParams):
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

    def fluxes(self, t, y) -> OpenFluxes:
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
        return OpenFluxes(z, u, z_t, M, p_s, p_g, p_a, V_s, V_g, A_w, m_surf, m_wall,
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
class OpenResult:
    model: OpenModel
    t: np.ndarray
    y: np.ndarray
    sol: object = field(repr=False)

    def flux_series(self, t=None):
        if t is None:
            t, y = self.t, self.y
        else:
            y = self.sol.sol(t)
        rows = [self.model.fluxes(ti, y[:, i]) for i, ti in enumerate(t)]
        out = {k: np.array([getattr(r, k) for r in rows]) for k in OpenFluxes.__dataclass_fields__}
        out["t"] = np.asarray(t)
        out["m_v"], out["T_v"], out["T_s"], out["T_w"] = y[2], y[3], y[4], y[5]
        return out

    def top_times(self, t_min=0.0):
        """Times of the top dead centre (u crosses 0 downwards), via dense output."""
        t = np.linspace(max(t_min, self.t[0]), self.t[-1], 40 * len(self.t))
        u = self.sol.sol(t)[1]
        i = np.where((u[:-1] > 0) & (u[1:] <= 0))[0]
        return t[i] - u[i] * (t[i + 1] - t[i]) / (u[i + 1] - u[i])


def simulate_open(p: OpenParams, t_end: float, z0=8.0, rtol=1e-8, max_step=0.01,
                  n_out=6000) -> OpenResult:
    m = OpenModel(p)
    y0 = m.initial_state(z0)
    atol = 1e-9 * (np.abs(y0) + 1e-3)
    sol = solve_ivp(m.rhs, (0, t_end), y0, method="Radau", rtol=rtol, atol=atol,
                    max_step=max_step, dense_output=True, t_eval=np.linspace(0, t_end, n_out))
    if not sol.success:
        raise RuntimeError(sol.message)
    return OpenResult(m, sol.t, sol.y, sol)


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
def open_stored_energy(res: OpenResult, fx):
    p = res.model.p
    KE = 0.5 * fx["M"] * fx["u"] ** 2
    PE = p.rho_l * G * (p.A_h * fx["z"] ** 2 + p.A_t * fx["z_t"] ** 2) / 2
    U_g = fx["p_g"] * fx["V_g"] / (p.gamma_g - 1)
    U_v = fx["m_v"] * wp.u_vapour(fx["T_v"])
    U_s = p.rho_l * p.area_factor * p.A_h * p.delta * wp.C_L * fx["T_s"]
    U_w = p.C_wall * p.A_wall_max * fx["T_w"] if p.wall_wet else 0.0 * fx["T_w"]
    return KE + PE + U_g + U_v + U_s + U_w


def open_boundary_rates(res: OpenResult, fx):
    p = res.model.p
    hl = wp.h_liquid
    W = p.A_h * fx["p_assist"] * fx["u"]
    cs, cw = np.minimum(fx["mdot_surf"], 0), np.minimum(fx["mdot_wall"], 0)
    Q = (fx["Q_bulk"] + fx["Q_core"] - fx["P_loss"]
         + cs * (hl(fx["T_s"]) - hl(p.T_bulk)) + cw * (hl(fx["T_w"]) - hl(p.T_bulk))
         + fx["mdot_flash"] * wp.h_fg(p.T_bulk)
         - fx["mdot_fog"] * (hl(fx["T_v"]) - hl(p.T_bulk)))
    H_mk = (fx["mdot_surf"] + fx["mdot_wall"] + fx["mdot_flash"] - fx["mdot_fog"]) * hl(p.T_bulk)
    H_out = fx["mdot_out"] * wp.h_vapour(fx["T_v"])
    return dict(W=W, Q=Q, H_mk=H_mk, H_out=H_out)


def open_energy_error(res: OpenResult, n=60000) -> float:
    t = np.linspace(res.t[0], res.t[-1], n)
    fx = res.flux_series(t)
    E = open_stored_energy(res, fx)
    r = open_boundary_rates(res, fx)
    integ = cumulative_trapezoid(r["W"] + r["Q"] + r["H_mk"] - r["H_out"], t, initial=0)
    KE = 0.5 * fx["M"] * fx["u"] ** 2
    scale = sum(np.trapezoid(np.abs(v), t) for v in r.values()) + np.ptp(KE) + 1e-12
    return float(np.max(np.abs((E - E[0]) - integ)) / scale)


@dataclass
class OpenCycle:
    period: float
    z_min: float
    z_max: float
    m_out: float
    m_flash: float
    m_surf_net: float
    m_wall_cond: float
    m_surf_cond: float
    H_net: float
    W: float
    W_loss: float
    Q: float
    exergy_out: float
    p_s_min: float
    p_s_max: float
    p_g_min: float
    p_g_max: float
    T_steam: float
    T_w_max: float
    closure: float

    @property
    def cop(self):
        """Hydram-style COP: net steam energy / drive work."""
        return self.H_net / self.W if self.W > 0 else np.nan

    @property
    def steam_power(self):
        return self.H_net / self.period

    @property
    def compression_ratio(self):
        return self.p_s_max / self.p_s_min

    @property
    def second_law_efficiency(self):
        return self.exergy_out / self.W if self.W > 0 else np.nan

    def summary(self):
        c = lambda T: T - 273.15
        return "\n".join([
            f"period                     {self.period:8.3f} s   (stroke {self.z_min:.2f} .. {self.z_max:.2f} m)",
            f"steam ejected / cycle      {self.m_out * 1e3:8.1f} g   ({self.m_out / self.period * 3600:.1f} kg/h)",
            f"flash / wall cond / surf cond {self.m_flash * 1e3:5.1f} / {self.m_wall_cond * 1e3:.1f} / {self.m_surf_cond * 1e3:.1f} g",
            f"net steam power            {self.steam_power / 1e3:8.2f} kW",
            f"drive work / cycle         {self.W / 1e3:8.2f} kJ  (hydraulic losses {self.W_loss / 1e3:.2f})",
            f"waste heat / cycle         {self.Q / 1e3:8.2f} kJ",
            f"COP (steam / drive work)   {self.cop:8.2f}",
            f"2nd-law efficiency         {self.second_law_efficiency * 100:8.1f} %",
            f"steam pressure             {self.p_s_min / 1e3:8.1f} .. {self.p_s_max / 1e3:.1f} kPa  (ratio {self.compression_ratio:.1f})",
            f"gas cushion pressure       {self.p_g_min / 1e3:8.1f} .. {self.p_g_max / 1e3:.1f} kPa",
            f"ejected steam temperature  {c(self.T_steam):8.1f} degC",
            f"peak wall temperature      {c(self.T_w_max):8.1f} degC",
            f"cycle energy closure       {self.closure:8.1e}",
        ])


def open_cycle_metrics(res: OpenResult, n=20000, which=-1) -> OpenCycle:
    """Metrics between consecutive top dead centres (``which=-1``: last cycle)."""
    p = res.model.p
    tops = res.top_times()
    if len(tops) < 2:
        raise RuntimeError("fewer than two complete cycles")
    t0, t1 = tops[which - 1], tops[which]
    t = np.linspace(t0, t1, n)
    fx = res.flux_series(t)
    r = open_boundary_rates(res, fx)
    I = lambda v: float(np.trapezoid(v, t))
    mo = fx["mdot_out"]
    m_out = I(mo)
    H_net = I(r["H_out"]) - m_out * wp.h_liquid(p.T_bulk)
    ex = mo * ((wp.h_vapour(fx["T_v"]) - wp.h_liquid(p.T_bulk))
               - p.T_bulk * (wp.s_vapour(fx["T_v"], fx["p_s"]) - wp.s_liquid(p.T_bulk)))
    E = open_stored_energy(res, fx)
    W, Q = I(r["W"]), I(r["Q"])
    resid = (E[-1] - E[0]) - (W + Q + I(r["H_mk"]) - I(r["H_out"]))
    return OpenCycle(
        period=t1 - t0, z_min=float(fx["z"].min()), z_max=float(fx["z"].max()),
        m_out=m_out, m_flash=I(fx["mdot_flash"]),
        m_surf_net=I(fx["mdot_surf"]),
        m_wall_cond=-I(np.minimum(fx["mdot_wall"], 0)),
        m_surf_cond=-I(np.minimum(fx["mdot_surf"], 0)),
        H_net=H_net, W=W, W_loss=I(fx["P_loss"]), Q=Q, exergy_out=I(ex),
        p_s_min=float(fx["p_s"].min()), p_s_max=float(fx["p_s"].max()),
        p_g_min=float(fx["p_g"].min()), p_g_max=float(fx["p_g"].max()),
        T_steam=I(mo * fx["T_v"]) / m_out if m_out > 0 else np.nan,
        T_w_max=float(fx["T_w"].max()),
        closure=resid / max(abs(H_net), abs(W), 1.0),
    )
