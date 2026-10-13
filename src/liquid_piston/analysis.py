"""Post-processing: conservation checks and per-cycle performance metrics.

The energy bookkeeping here is written independently of the RHS in
``model.py``: stored energy is computed from the states, and the boundary
fluxes from their own formulas. If the two disagree, there is a sign or
reference error somewhere, which is exactly what the closure test catches.

System boundary: liquid column + vapour chamber + surface layer + bulk
liquid. The bulk is held at ``T_bulk`` by the waste-heat supply and topped
up with make-up water at ``T_bulk`` (so its mass stays constant).

    dE/dt = W_drive + Q_waste + H_makeup - H_steam

    E         = KE + PE + U_vapour + U_layer
    W_drive   = A p2 v                          (mechanical work on leg 2)
    Q_waste   = heat the bulk must receive to stay at T_bulk
    H_makeup  = (mdot_evap + mdot_flash - mdot_fog) * h_l(T_bulk)
    H_steam   = mdot_out * h_v(T_v)

Friction heat is deposited in the bulk liquid, so it reduces ``Q_waste``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import properties as wp
from .model import G, Result


def stored_energy(res: Result, fx: dict) -> np.ndarray:
    p = res.model.p
    KE = 0.5 * p.m_liquid * fx["v"] ** 2
    PE = p.rho_l * G * p.A * fx["x"] ** 2
    if p.chamber == "open":
        return KE + PE
    U_v = fx["m_v"] * wp.u_vapour(fx["T_v"])
    U_s = p.m_layer * wp.h_liquid(fx["T_s"])
    return KE + PE + U_v + U_s


def boundary_rates(res: Result, fx: dict) -> dict:
    p = res.model.p
    me = fx["mdot_evap"]
    W = p.A * fx["p2"] * fx["v"]
    if p.chamber == "open":
        # plain U-tube: the constant pressure above leg 1 also does work,
        # and friction heat leaves the (purely mechanical) system
        W = W - p.A * fx["p_v"] * fx["v"]
        zero = np.zeros_like(W)
        return dict(W=W, Q=-fx["P_fric"], H_mk=zero, H_out=zero)
    cond = np.minimum(me, 0.0)
    mf, mg = fx["mdot_flash"], fx["mdot_fog"]
    Q = (fx["Q_bulk"] + fx["Q_wall"] - fx["P_fric"]
         + cond * (wp.h_liquid(fx["T_s"]) - wp.h_liquid(p.T_bulk))
         + mf * wp.h_fg(p.T_bulk)
         - mg * (wp.h_liquid(fx["T_v"]) - wp.h_liquid(p.T_bulk)))
    H_mk = (me + mf - mg) * wp.h_liquid(p.T_bulk)
    H_out = fx["mdot_out"] * wp.h_vapour(fx["T_v"])
    return dict(W=W, Q=Q, H_mk=H_mk, H_out=H_out)


@dataclass
class Balance:
    t: np.ndarray
    dE: np.ndarray           # change of stored energy from the states, J
    integrated: np.ndarray   # time integral of the boundary rates, J
    dm: np.ndarray           # change of vapour mass, kg
    m_net: np.ndarray        # integral of (evaporation - ejection), kg
    throughput: float        # scale for relative errors, J

    @property
    def energy_error(self) -> float:
        return float(np.max(np.abs(self.dE - self.integrated)) / self.throughput)

    @property
    def mass_error(self) -> float:
        scale = max(np.max(np.abs(self.m_net)), 1e-12)
        return float(np.max(np.abs(self.dm - self.m_net)) / scale)


def balances(res: Result, n: int = 40000) -> Balance:
    t = np.linspace(res.t[0], res.t[-1], n)
    fx = res.flux_series(t)
    E = stored_energy(res, fx)
    r = boundary_rates(res, fx)
    total = r["W"] + r["Q"] + r["H_mk"] - r["H_out"]
    integ = cumulative_trapezoid(total, t, initial=0.0)
    throughput = sum(np.trapezoid(np.abs(v), t) for v in r.values()) + 1e-12
    if res.model.p.chamber == "open":
        throughput = max(throughput, float(np.max(E)))
    m_net = cumulative_trapezoid(fx["mdot_evap"] + fx["mdot_flash"] - fx["mdot_fog"]
                                 - fx["mdot_out"], t, initial=0.0)
    return Balance(t, E - E[0], integ, fx["m_v"] - fx["m_v"][0], m_net, throughput)


@dataclass
class CycleMetrics:
    period: float
    m_out: float            # steam ejected per cycle, kg
    m_evap: float           # gross free-surface evaporation per cycle, kg
    m_flash: float          # bulk flash boiling per cycle, kg
    m_fog: float            # homogeneous condensation per cycle, kg
    m_cond: float           # gross condensation per cycle, kg
    H_net: float            # steam enthalpy minus make-up enthalpy, J
    W: float                # drive work per cycle, J
    W_pv: float             # vapour compression work (closed integral of p dV), J
    W_fric: float           # friction dissipation per cycle, J
    Q: float                # waste heat per cycle, J
    exergy_out: float       # exergy added to the water stream, J
    p_max: float            # peak chamber pressure, Pa
    p_min: float            # minimum chamber pressure, Pa
    T_v_max: float          # peak vapour temperature, K
    T_steam: float          # mass-averaged temperature of ejected steam, K
    T_s_max: float          # peak surface-layer temperature, K
    min_superheat: float    # min of T_v - T_sat(p_v); < 0 means the vapour is
                            # supersaturated (bulk fogging is not modelled)
    p2_amp: float           # half peak-to-peak drive pressure, Pa
    closure: float          # (W + Q - H_net) / H_net over the cycle

    @property
    def steam_power(self) -> float:
        return self.H_net / self.period

    @property
    def steam_rate(self) -> float:
        return self.m_out / self.period

    @property
    def work_share(self) -> float:
        """Fraction of the delivered steam energy supplied as mechanical work."""
        return self.W / self.H_net if self.H_net > 0 else np.nan

    @property
    def second_law_efficiency(self) -> float:
        """Exergy delivered to the stream / drive work. Must be <= 1."""
        return self.exergy_out / self.W if self.W > 0 else np.nan

    def summary(self) -> str:
        c = lambda T: T - 273.15
        return "\n".join([
            f"period                    {self.period:8.3f} s",
            f"steam ejected / cycle     {self.m_out * 1e3:8.2f} g  ({self.steam_rate * 3600:.1f} kg/h)",
            f"flash / evap / cond / fog {self.m_flash * 1e3:8.1f} / {self.m_evap * 1e3:.1f} / {self.m_cond * 1e3:.1f} / {self.m_fog * 1e3:.1f} g",
            f"steam power (net)         {self.steam_power / 1e3:8.2f} kW",
            f"drive work / cycle        {self.W / 1e3:8.2f} kJ  (pV {self.W_pv / 1e3:.2f}, friction {self.W_fric / 1e3:.2f})",
            f"waste heat / cycle        {self.Q / 1e3:8.2f} kJ",
            f"work share of steam energy{self.work_share * 100:8.1f} %",
            f"2nd-law efficiency        {self.second_law_efficiency * 100:8.1f} %",
            f"chamber pressure          {self.p_min / 1e3:8.1f} .. {self.p_max / 1e3:.1f} kPa",
            f"ejected steam temperature {c(self.T_steam):8.1f} degC (peak vapour {c(self.T_v_max):.1f} degC)",
            f"peak surface temperature  {c(self.T_s_max):8.1f} degC",
            f"min superheat             {self.min_superheat:8.1f} K",
            f"drive pressure amplitude  {self.p2_amp / 1e3:8.1f} kPa",
            f"cycle energy closure      {self.closure:8.1e}",
        ])


def cycle_metrics(res: Result, n: int = 20000) -> CycleMetrics:
    """Metrics over the last full drive period (call after a periodic state)."""
    drive = res.model.drive
    p = res.model.p
    T = 1.0 / drive.freq
    t = np.linspace(res.t[-1] - T, res.t[-1], n)
    fx = res.flux_series(t)
    r = boundary_rates(res, fx)
    I = lambda y: float(np.trapezoid(y, t))
    me, mo = fx["mdot_evap"], fx["mdot_out"]

    m_out = I(mo)
    H_net = I(r["H_out"]) - I(mo) * wp.h_liquid(p.T_bulk)
    s_out = wp.s_vapour(fx["T_v"], fx["p_v"])
    s_mk = wp.s_liquid(p.T_bulk)
    ex = mo * ((wp.h_vapour(fx["T_v"]) - wp.h_liquid(p.T_bulk)) - p.T_bulk * (s_out - s_mk))
    W = I(r["W"])
    Q = I(r["Q"])
    # cycle closure: in a periodic state, W + Q + H_mk - H_out ~ 0, and
    # make-up equals net evaporation (= ejected mass when periodic)
    E_res = W + Q + I(r["H_mk"]) - I(r["H_out"])
    T_steam = I(mo * fx["T_v"]) / m_out if m_out > 0 else np.nan

    return CycleMetrics(
        period=T,
        m_out=m_out,
        m_evap=I(np.maximum(me, 0.0)),
        m_flash=I(fx["mdot_flash"]),
        m_fog=I(fx["mdot_fog"]),
        m_cond=-I(np.minimum(me, 0.0)),
        H_net=H_net,
        W=W,
        W_pv=I(p.A * fx["p_v"] * fx["v"]),
        W_fric=I(fx["P_fric"]),
        Q=Q,
        exergy_out=I(ex),
        p_max=float(fx["p_v"].max()),
        p_min=float(fx["p_v"].min()),
        T_v_max=float(fx["T_v"].max()),
        T_steam=T_steam,
        T_s_max=float(fx["T_s"].max()),
        min_superheat=float(np.min(fx["T_v"] - wp.T_sat(fx["p_v"]))),
        p2_amp=float(0.5 * (fx["p2"].max() - fx["p2"].min())),
        closure=E_res / H_net if H_net > 0 else np.nan,
    )


def run_to_periodic(params, drive, n_cycles: int = 20, **kw) -> Result:
    """Integrate ``n_cycles`` drive periods from the saturated initial state."""
    from .model import simulate
    return simulate(params, drive, n_cycles / drive.freq, **kw)
