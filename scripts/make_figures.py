"""Generate the figures in ``figures/`` (run ``run_study.py`` first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liquid_piston import (REFERENCE, KinematicDrive, Params, PressureDrive,  # noqa: E402
                           balances, reference_drive, run_to_periodic, simulate)
from liquid_piston import properties as wp  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
TEAL, AMBER, SLATE, RED, GREY = "#0F766E", "#B45309", "#334155", "#B91C1C", "#94A3B8"
C = lambda T: np.asarray(T) - 273.15

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E2E8F0", "grid.linewidth": 0.6, "legend.frameon": False,
})


def last_cycle(res, n=2000):
    T = 1.0 / res.model.drive.freq
    t = np.linspace(res.t[-1] - T, res.t[-1], n)
    return (t - t[0]) / T, res.flux_series(t)


def fig_cycle(res):
    s, fx = last_cycle(res)
    p = REFERENCE
    fig, ax = plt.subplots(4, 1, figsize=(6.4, 7.6), sharex=True)
    ax[0].plot(s, p.H0 - fx["x"], color=SLATE)
    ax[0].set_ylabel("headspace height (m)")
    ax[0].set_title("One periodic cycle of the reference case")

    ax[1].plot(s, fx["p_v"] / 1e3, color=TEAL, label="chamber")
    ax[1].axhline(p.p_out / 1e3, color=AMBER, ls="--", lw=1, label="outlet (valve opens)")
    ax[1].axhline(wp.p_sat(p.T_bulk) / 1e3, color=GREY, ls=":", lw=1, label="p_sat(T_bulk)")
    ax[1].set_ylabel("pressure (kPa)")
    ax[1].legend(loc="upper right")

    ax[2].plot(s, C(fx["T_v"]), color=TEAL, label="vapour")
    ax[2].plot(s, C(fx["T_s"]), color=AMBER, label="surface layer")
    ax[2].plot(s, C(wp.T_sat(fx["p_v"])), color=GREY, ls=":", label="T_sat(p_v)")
    ax[2].axhline(C(p.T_bulk), color=SLATE, lw=0.8, ls="--", label="bulk")
    ax[2].set_ylabel("temperature (degC)")
    ax[2].legend(loc="upper right", ncol=2)

    ax[3].plot(s, fx["mdot_flash"], color=TEAL, label="flash boiling")
    ax[3].plot(s, fx["mdot_evap"], color=AMBER, label="free surface (+evap / -cond)")
    ax[3].plot(s, fx["mdot_out"], color=RED, label="ejected")
    ax[3].plot(s, -fx["mdot_fog"], color=GREY, label="fog (-)")
    ax[3].set_ylabel("mass flow (kg/s)")
    ax[3].set_xlabel("time / period")
    ax[3].legend(loc="upper right", ncol=2)
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(FIG / "cycle.png")
    plt.close(fig)


def fig_pv(res):
    s, fx = last_cycle(res, 4000)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ej = fx["mdot_out"] > 0
    ax.loglog(fx["V"], fx["p_v"] / 1e3, color=TEAL, lw=1.4, label="cycle")
    ax.loglog(fx["V"][ej], fx["p_v"][ej] / 1e3, ".", ms=2, color=RED, label="valve open")
    i0 = int(np.argmax(fx["V"]))
    V = np.linspace(fx["V"].min(), fx["V"].max(), 100)
    ax.loglog(V, fx["p_v"][i0] / 1e3 * (fx["V"][i0] / V) ** wp.GAMMA_V, color=GREY, ls=":",
              label="isentrope from max volume")
    ax.axhline(REFERENCE.p_out / 1e3, color=AMBER, ls="--", lw=1, label="outlet pressure")
    ax.set_yticks([20, 50, 100, 150, 200, 400], ["20", "50", "100", "150", "200", "400"])
    ax.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.annotate("compression", xy=(0.9, 60), xytext=(0.45, 30), fontsize=8, color=SLATE,
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8))
    ax.set_xlabel("vapour volume (m$^3$)")
    ax.set_ylabel("chamber pressure (kPa)")
    ax.set_title("p-V diagram of the vapour chamber")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "pv.png")
    plt.close(fig)


def fig_verification():
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
    P_ATM = 101.325e3

    # rung 1: frictionless U-tube, energy error
    p1 = Params(chamber="open", p_open=P_ATM, K_loss=0.0)
    T1 = 2 * np.pi / p1.omega_hydrostatic
    r1 = simulate(p1, PressureDrive(P_ATM), 10 * T1, x0=0.5)
    b1 = balances(r1)
    err = np.abs(b1.dE - b1.integrated) / b1.throughput
    k = np.floor(b1.t / T1).astype(int)
    per = [err[k == j].max() for j in range(10)]
    ax[0, 0].bar(np.arange(10) + 0.5, per, width=0.7, color=TEAL)
    ax[0, 0].set_title("1. Frictionless U-tube: max energy error per period")
    ax[0, 0].set_xlabel("time / period")
    ax[0, 0].set_ylabel("relative error")
    ax[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # rung 2: quadratic friction decay vs analytic envelope
    p2 = Params(chamber="open", p_open=P_ATM, K_loss=3.0)
    w = p2.omega_hydrostatic
    r2 = simulate(p2, PressureDrive(P_ATM), 12 * T1, x0=0.5, n_out=20000)
    c = p2.K_loss / (2 * p2.L)
    tt = np.linspace(0, r2.t[-1], 400)
    ax[0, 1].plot(r2.t / T1, r2.y[0], color=SLATE, lw=0.8, label="model")
    env = 0.5 / (1 + 4 / (3 * np.pi) * c * w * 0.5 * tt)
    ax[0, 1].plot(tt / T1, env, color=AMBER, ls="--", label="analytic envelope")
    ax[0, 1].plot(tt / T1, -env, color=AMBER, ls="--")
    ax[0, 1].set_title("2. Quadratic friction: amplitude decay")
    ax[0, 1].set_xlabel("time / period")
    ax[0, 1].set_ylabel("x (m)")
    ax[0, 1].legend(loc="upper right")

    # rung 3: gas spring, p V^gamma invariant
    p3 = Params(sigma=0.0, valve_area=0.0, K_loss=0.0)
    p0 = float(wp.p_sat(p3.T_bulk))
    r3 = simulate(p3, PressureDrive(p0), 30.0, x0=0.8, n_out=6000)
    fx3 = r3.flux_series()
    inv = fx3["p_v"] * fx3["V"] ** wp.GAMMA_V
    ax[1, 0].plot(r3.t, inv / inv[0] - 1, color=TEAL)
    ax[1, 0].set_title("3. Gas spring: drift of p V$^\\gamma$")
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel("relative drift")
    ax[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # rung 4: relaxation to saturation
    p4 = Params(valve_area=0.0, h_wall=50.0)
    ps = float(wp.p_sat(p4.T_bulk))
    r4 = simulate(p4, KinematicDrive(0.0, 0.0, 0.2), 600.0, p_v0=0.5 * ps, n_out=3000)
    fx4 = r4.flux_series()
    ax[1, 1].plot(r4.t, fx4["p_v"] / 1e3, color=TEAL, label="chamber")
    ax[1, 1].axhline(ps / 1e3, color=AMBER, ls="--", label="p_sat(T_bulk)")
    ax[1, 1].set_xscale("symlog", linthresh=1.0)
    ax[1, 1].set_xlim(0, 600)
    ax[1, 1].set_title("4. Still chamber relaxes to saturation")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylabel("pressure (kPa)")
    ax[1, 1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "verification.png")
    plt.close(fig)


def fig_sensitivity(sweeps):
    names = ["T_bulk", "stroke", "h_wall", "delta", "h_bulk", "cav", "freq", "sigma"]
    fig, axs = plt.subplots(4, 2, figsize=(7.2, 10.0))
    for ax, name in zip(axs.flat, names):
        sw = sweeps[name]
        x = [r["value"] for r in sw["rows"]]
        P = [r["steam_power"] / 1e3 for r in sw["rows"]]
        ws = [np.nan if r["work_share"] is None else 100 * r["work_share"] for r in sw["rows"]]
        ax.plot(x, P, "o-", color=TEAL, ms=4)
        ax.set_ylabel("net steam power (kW)", color=TEAL)
        ax.set_xlabel(f"{sw['label']} ({sw['unit']})")
        ax2 = ax.twinx()
        ax2.plot(x, ws, "s--", color=AMBER, ms=3.5)
        ax2.set_ylabel("work share (%)", color=AMBER)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ref = {"T_bulk": 60, "stroke": 1.3, "h_wall": 0, "delta": 1.0, "h_bulk": 2000,
               "cav": 10, "freq": 1.0, "sigma": 0.04}[name]
        ax.axvline(ref, color=GREY, lw=0.8, ls=":")
    axs[2, 0].set_xscale("log")
    axs[2, 1].set_xscale("symlog", linthresh=1.0)
    axs[2, 1].set_xlim(-0.2, 40)
    axs[2, 1].set_xticks([0, 1, 3, 10, 30], ["0", "1", "3", "10", "30"])
    axs[3, 1].set_xscale("log")
    fig.suptitle("One-at-a-time sensitivity (dotted line: reference case)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png")
    plt.close(fig)


def fig_frequency(sweeps):
    sw = sweeps["freq"]
    x = [r["value"] for r in sw["rows"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(x, [r["p2_amp"] / 1e3 for r in sw["rows"]], "o-", color=TEAL, label="drive pressure amplitude")
    ax.set_xlabel("drive frequency / plain U-tube frequency")
    ax.set_ylabel("required drive pressure (kPa)", color=TEAL)
    ax2 = ax.twinx()
    ax2.plot(x, [r["W"] / r["period"] / 1e3 for r in sw["rows"]], "s--", color=AMBER, label="drive power")
    ax2.set_ylabel("drive power (kW)", color=AMBER)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Same stroke at different frequencies")
    fig.tight_layout()
    fig.savefig(FIG / "frequency.png")
    plt.close(fig)


def fig_energy(ref):
    kJ = lambda v: v / 1e3
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    W_pv, W_f, Q, H = kJ(ref["W_pv"]), kJ(ref["W_fric"]), kJ(ref["Q"]), kJ(ref["H_net"])
    ax.barh(1, W_pv, color=SLATE, label=f"compression work {W_pv:.0f} kJ")
    ax.barh(1, W_f, left=W_pv, color=GREY, label=f"friction {W_f:.0f} kJ")
    ax.barh(1, Q, left=W_pv + W_f, color=AMBER, label=f"waste heat {Q:.0f} kJ")
    ax.barh(0, H, color=TEAL, label=f"net steam enthalpy {H:.0f} kJ")
    ax.set_yticks([0, 1], ["out", "in"])
    ax.set_xlabel("energy per cycle (kJ)")
    ax.set_title("First law over one cycle (friction heat returns to the bulk)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "energy.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    res = run_to_periodic(REFERENCE, reference_drive(), n_cycles=15)
    fig_cycle(res)
    fig_pv(res)
    fig_verification()
    sweeps = json.loads((RES / "sweeps.json").read_text())
    ref = json.loads((RES / "reference.json").read_text())
    fig_sensitivity(sweeps)
    fig_energy(ref)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
