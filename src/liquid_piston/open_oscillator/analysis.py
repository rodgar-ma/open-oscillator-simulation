"""Energy bookkeeping and per-cycle metrics for the open oscillator.

As for the U-tube model, the energy check is written independently of the
right-hand side: stored energy comes from the states and boundary fluxes from
their own formulas, so a sign or reference error shows up as a closure error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .. import properties as wp
from .model import G, Result


def stored_energy(res: Result, fx):
    p = res.model.p
    KE = 0.5 * fx["M"] * fx["u"] ** 2
    PE = p.rho_l * G * (p.A_h * fx["z"] ** 2 + p.A_t * fx["z_t"] ** 2) / 2
    U_g = fx["p_g"] * fx["V_g"] / (p.gamma_g - 1)
    U_v = fx["m_v"] * wp.u_vapour(fx["T_v"])
    U_s = p.rho_l * p.area_factor * p.A_h * p.delta * wp.C_L * fx["T_s"]
    U_w = p.C_wall * p.A_wall_max * fx["T_w"] if p.wall_wet else 0.0 * fx["T_w"]
    return KE + PE + U_g + U_v + U_s + U_w


def boundary_rates(res: Result, fx):
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


def energy_error(res: Result, n=60000) -> float:
    t = np.linspace(res.t[0], res.t[-1], n)
    fx = res.flux_series(t)
    E = stored_energy(res, fx)
    r = boundary_rates(res, fx)
    integ = cumulative_trapezoid(r["W"] + r["Q"] + r["H_mk"] - r["H_out"], t, initial=0)
    KE = 0.5 * fx["M"] * fx["u"] ** 2
    scale = sum(np.trapezoid(np.abs(v), t) for v in r.values()) + np.ptp(KE) + 1e-12
    return float(np.max(np.abs((E - E[0]) - integ)) / scale)


@dataclass
class CycleMetrics:
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


def cycle_metrics(res: Result, n=20000, which=-1) -> CycleMetrics:
    """Metrics between consecutive top dead centres (``which=-1``: last cycle)."""
    p = res.model.p
    tops = res.top_times()
    if len(tops) < 2:
        raise RuntimeError("fewer than two complete cycles")
    t0, t1 = tops[which - 1], tops[which]
    t = np.linspace(t0, t1, n)
    fx = res.flux_series(t)
    r = boundary_rates(res, fx)
    I = lambda v: float(np.trapezoid(v, t))
    mo = fx["mdot_out"]
    m_out = I(mo)
    H_net = I(r["H_out"]) - m_out * wp.h_liquid(p.T_bulk)
    ex = mo * ((wp.h_vapour(fx["T_v"]) - wp.h_liquid(p.T_bulk))
               - p.T_bulk * (wp.s_vapour(fx["T_v"], fx["p_s"]) - wp.s_liquid(p.T_bulk)))
    E = stored_energy(res, fx)
    W, Q = I(r["W"]), I(r["Q"])
    resid = (E[-1] - E[0]) - (W + Q + I(r["H_mk"]) - I(r["H_out"]))
    return CycleMetrics(
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
