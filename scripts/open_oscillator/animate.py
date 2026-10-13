"""Animated simulation of the open oscillator (high pipe + side tank).

Architecture as described publicly by Hydram Dynamics: a liquid slug
oscillates between a tall high pipe, whose top holds the steam, and a wide
side tank with a gas cushion; a small assist pressure in the direction of
travel sustains the oscillation. Parameters are illustrative.

    python scripts/open_oscillator/animate.py                      # MP4, reference case
    python scripts/open_oscillator/animate.py --out animation.gif  # GIF
    python scripts/open_oscillator/animate.py --h-cond 2000        # strongly wetted walls
    python scripts/open_oscillator/animate.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from liquid_piston import properties as wp  # noqa: E402
from liquid_piston.open_oscillator import REFERENCE, cycle_metrics, simulate  # noqa: E402

TEAL, AMBER, SLATE, RED, GREY, PURPLE = "#0F766E", "#B45309", "#334155", "#B91C1C", "#94A3B8", "#7C3AED"
WATER, GAS = "#2563EB", "#CBD5E1"
T_MIN_C, T_MAX_C = 40.0, 300.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-bulk", type=float, default=REFERENCE.T_bulk - 273.15, help="waste-heat temperature, degC")
    ap.add_argument("--assist", type=float, default=REFERENCE.P_assist / 1e3, help="drive assist pressure, kPa")
    ap.add_argument("--h-cond", type=float, default=REFERENCE.h_cond, help="wall condensation coefficient, W/m2K")
    ap.add_argument("--t-end", type=float, default=60.0, help="simulated time before the animated cycles, s")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "open_oscillator" / "animation.mp4")
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


def prepare(args):
    p = REFERENCE.with_(T_bulk=273.15 + args.t_bulk, P_assist=args.assist * 1e3, h_cond=args.h_cond)
    res = simulate(p, args.t_end)
    m = cycle_metrics(res)
    tops = res.top_times()
    t0, t1 = tops[-1 - args.cycles], tops[-1]
    n = int(round((t1 - t0) * args.fps / args.speed))
    t = np.linspace(t0, t1, n)
    fx = res.flux_series(t)
    fx["tau"] = t - t0
    return p, fx, m


def phase_label(f, i):
    if f["mdot_out"][i] > 0:
        return "3. Ejection: valve open", RED
    if f["u"][i] > 0:
        return "2. Compression by the liquid piston", AMBER
    if f["mdot_flash"][i] > 0:
        return "1. Flash evaporation behind the falling slug", TEAL
    return "Fall: residual steam expands", SLATE


class Schematic:
    def __init__(self, ax, p, cmap, norm):
        self.ax, self.p, self.cmap, self.norm = ax, p, cmap, norm
        Dh = p.D_h
        self.xh = (-Dh / 2, Dh / 2)
        self.x_tank = (1.6, 1.6 + p.D_t)
        self.y_bot = -Dh
        self.tank_top = p.z_t0 + p.V_g0 / p.A_t
        self.valve = (0.0, p.H_top + 0.35)
        self.outlet = (0.0, p.H_top + 0.9)

        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 3.9)
        ax.set_ylim(self.y_bot - 0.6, p.H_top + 3.6)
        ax.axis("off")

        self.bottom = Rectangle((self.xh[0], self.y_bot), self.x_tank[1] - self.xh[0], -self.y_bot,
                                color=WATER, alpha=0.85, lw=0)
        self.liq_h = Rectangle((self.xh[0], 0), Dh, 1, color=WATER, alpha=0.85, lw=0)
        self.liq_t = Rectangle((self.x_tank[0], 0), p.D_t, 1, color=WATER, alpha=0.85, lw=0)
        self.gas = Rectangle((self.x_tank[0], 0), p.D_t, 1, color=GAS, lw=0)
        self.vap = Rectangle((self.xh[0], 0), Dh, 1, lw=0)
        self.layer = Rectangle((self.xh[0], 0), Dh, 0.12, lw=0)
        for patch in (self.bottom, self.liq_h, self.liq_t, self.gas, self.vap, self.layer):
            ax.add_patch(patch)

        wall = dict(color=SLATE, lw=2.0, solid_capstyle="round")
        ax.plot([self.xh[0]] * 2, [self.y_bot, p.H_top], **wall)
        ax.plot([self.xh[1]] * 2, [0, p.H_top], **wall)
        ax.plot(list(self.xh), [p.H_top] * 2, **wall)
        ax.plot([self.xh[1], self.x_tank[0]], [0, 0], **wall)
        ax.plot([self.x_tank[0]] * 2, [0, self.tank_top], **wall)
        ax.plot([self.x_tank[1]] * 2, [self.y_bot, self.tank_top], **wall)
        ax.plot(list(self.x_tank), [self.tank_top] * 2, **wall)
        ax.plot([self.xh[0], self.x_tank[1]], [self.y_bot] * 2, **wall)

        # waste-heat exchanger wrapped around the top of the high pipe
        for side in (self.xh[0] - 0.22, self.xh[1] + 0.02):
            ax.add_patch(Rectangle((side, p.H_top - 3.2), 0.2, 3.0, fc="#FDE68A", ec=AMBER, lw=1))
        ax.text(self.xh[0] - 0.3, p.H_top - 1.7, "waste-heat\nexchanger", ha="right", va="center",
                fontsize=7, color=AMBER)

        ax.plot([0, 0], [p.H_top, self.outlet[1]], color=SLATE, lw=2.0)
        vx, vy = self.valve
        self.valve_patch = Polygon([[vx - 0.2, vy - 0.14], [vx + 0.2, vy + 0.14],
                                    [vx + 0.2, vy - 0.14], [vx - 0.2, vy + 0.14]],
                                   closed=True, fc="white", ec=SLATE, lw=1.4, zorder=5)
        ax.add_patch(self.valve_patch)
        ax.text(vx - 0.3, vy, "ejection valve", fontsize=7, color=SLATE, va="center", ha="right")

        ax.text(0, self.y_bot - 0.2, "high pipe", ha="center", va="top", fontsize=7)
        ax.text(np.mean(self.x_tank), self.y_bot - 0.2, "side tank", ha="center", va="top", fontsize=7)
        self.gas_text = ax.text(np.mean(self.x_tank), self.tank_top + 0.25, "", ha="center", fontsize=7.5)
        self.assist = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2))
        ax.text(self.x_tank[1] - 0.1, self.tank_top + 1.6, "drive\nassist", fontsize=7, color=PURPLE)
        self.info = ax.text(1.0, p.H_top - 0.4, "", fontsize=8, family="monospace", va="bottom",
                            bbox=dict(fc="white", ec="#CBD5E1", boxstyle="round,pad=0.4"))
        self.phase = ax.text(-2.55, p.H_top + 3.5, "", fontsize=10.5, weight="bold", va="top")

        self.rng = np.random.default_rng(3)
        self.bubbles = np.empty((0, 2))
        self.steam = np.empty((0, 3))
        self.bubble_sc = ax.scatter([], [], s=8, facecolors="white", edgecolors="#BFDBFE",
                                    linewidths=0.5, zorder=4)
        self.steam_sc = ax.scatter([], [], s=50, color="#E2E8F0", edgecolors="none", zorder=6)

    def update(self, f, i, dt, mf_max, mo_max):
        p = self.p
        z, zt = f["z"][i], f["z_t"][i]
        self.liq_h.set_height(z)
        self.vap.set_y(z)
        self.vap.set_height(p.H_top - z)
        self.vap.set_color(self.cmap(self.norm(f["T_v"][i] - 273.15)))
        self.layer.set_y(z - 0.12)
        self.layer.set_color(self.cmap(self.norm(f["T_s"][i] - 273.15)))
        self.liq_t.set_height(zt)
        self.gas.set_y(zt)
        self.gas.set_height(self.tank_top - zt)
        self.gas_text.set_text(f"gas {f['p_g'][i] / 1e3:4.0f} kPa")

        # assist pushes on the gas cushion in the slug's direction of travel
        cx, cy = np.mean(self.x_tank), self.tank_top + 1.45
        k = f["p_assist"][i] / max(p.P_assist, 1.0)
        self.assist.xy = (cx, cy - 0.55 * k)
        self.assist.set_position((cx, cy))
        self.assist.set_visible(abs(k) > 0.05)

        self.valve_patch.set_facecolor(RED if f["mdot_out"][i] > 0 else "white")

        n = self.rng.poisson(12 * f["mdot_flash"][i] / mf_max) if mf_max > 0 else 0
        if n:
            new = np.column_stack([self.rng.uniform(self.xh[0] + 0.05, self.xh[1] - 0.05, n),
                                   z - self.rng.uniform(0.05, 1.5, n)])
            self.bubbles = np.vstack([self.bubbles, new])
        if len(self.bubbles):
            self.bubbles[:, 1] += 2.0 * dt
            self.bubbles = self.bubbles[self.bubbles[:, 1] < z - 0.03]
        self.bubble_sc.set_offsets(self.bubbles if len(self.bubbles) else np.empty((0, 2)))

        n = self.rng.poisson(8 * f["mdot_out"][i] / mo_max) if mo_max > 0 else 0
        if n:
            ox, oy = self.outlet
            new = np.column_stack([ox + self.rng.normal(0, 0.05, n), oy + self.rng.uniform(0, 0.1, n),
                                   np.zeros(n)])
            self.steam = np.vstack([self.steam, new])
        if len(self.steam):
            self.steam[:, 0] += self.rng.normal(0.3, 0.25, len(self.steam)) * dt
            self.steam[:, 1] += 2.0 * dt
            self.steam[:, 2] += dt
            self.steam = self.steam[self.steam[:, 2] < 1.2]
        if len(self.steam):
            self.steam_sc.set_offsets(self.steam[:, :2])
            a = np.clip(1 - self.steam[:, 2] / 1.2, 0, 1)
            self.steam_sc.set_facecolors([(0.85, 0.88, 0.92, x) for x in a])
            self.steam_sc.set_sizes(40 + 160 * self.steam[:, 2])
        else:
            self.steam_sc.set_offsets(np.empty((0, 2)))

        label, color = phase_label(f, i)
        self.phase.set_text(label)
        self.phase.set_color(color)
        Tsat = wp.T_sat(f["p_s"][i]) - 273.15
        self.info.set_text(
            f"steam  p    {f['p_s'][i] / 1e3:6.1f} kPa\n"
            f"       T    {f['T_v'][i] - 273.15:6.1f} degC\n"
            f"       Tsat {Tsat:6.1f} degC\n"
            f"slug   z    {z:6.2f} m\n"
            f"       u    {f['u'][i]:6.2f} m/s")


def build_figure(p, f, m):
    cmap, norm = plt.get_cmap("plasma"), Normalize(T_MIN_C, T_MAX_C)
    fig = plt.figure(figsize=(12.6, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], hspace=0.45, wspace=0.36,
                          left=0.01, right=0.98, top=0.88, bottom=0.08)
    ax_s = fig.add_subplot(gs[:, 0])
    ax_p, ax_pv = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_T = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_m = fig.add_subplot(gs[1, 2], sharex=ax_p)
    fig.suptitle(
        f"Open oscillator (high pipe + side tank), 0D model  |  waste heat {p.T_bulk - 273.15:.0f} degC, "
        f"pipe D {p.D_h:.1f} m x {p.H_top:.0f} m, outlet {p.p_out / 1e3:.0f} kPa\n"
        f"per cycle ({m.period:.2f} s): {m.m_out * 1e3:.0f} g steam, {m.steam_power / 1e3:.0f} kW net, "
        f"COP {m.cop:.2f}, compression ratio {m.compression_ratio:.1f}   |   illustrative parameters, not Hydram's design",
        fontsize=10)
    sch = Schematic(ax_s, p, cmap, norm)
    cax = ax_s.inset_axes([0.06, 0.08, 0.03, 0.3])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("steam / surface (degC)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    tau = f["tau"]
    ax_p.plot(tau, f["p_s"] / 1e3, color=TEAL, lw=1.4, label="steam")
    ax_p.plot(tau, f["p_g"] / 1e3, color=PURPLE, lw=1.1, label="gas cushion")
    ax_p.axhline(p.p_out / 1e3, color=AMBER, ls="--", lw=1, label="outlet")
    ax_p.set_ylabel("pressure (kPa)", fontsize=8)
    ax_p.set_title("Pressures", fontsize=9)
    ax_p.legend(fontsize=7, frameon=False, loc="upper right")

    ax_pv.loglog(f["V_s"], f["p_s"] / 1e3, color=TEAL, lw=1.2)
    ax_pv.axhline(p.p_out / 1e3, color=AMBER, ls="--", lw=1)
    ax_pv.set_xlabel("steam volume (m$^3$)", fontsize=8)
    ax_pv.set_ylabel("pressure (kPa)", fontsize=8)
    ax_pv.set_title("p-V diagram of the steam space", fontsize=9)
    ax_pv.minorticks_off()
    ax_pv.set_yticks([20, 50, 100, 150], ["20", "50", "100", "150"])
    ax_pv.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])

    ax_T.plot(tau, f["T_v"] - 273.15, color=TEAL, lw=1.4, label="steam")
    ax_T.plot(tau, f["T_s"] - 273.15, color=AMBER, lw=1.2, label="free surface")
    ax_T.plot(tau, f["T_w"] - 273.15, color=SLATE, lw=1.0, label="wall")
    ax_T.plot(tau, wp.T_sat(f["p_s"]) - 273.15, color=GREY, ls=":", lw=1, label="T_sat(p)")
    ax_T.set_ylabel("temperature (degC)", fontsize=8)
    ax_T.set_xlabel("time (s)", fontsize=8)
    ax_T.set_title("Temperatures", fontsize=9)
    ax_T.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)

    ax_m.plot(tau, f["mdot_flash"], color=TEAL, lw=1.2, label="flash")
    ax_m.plot(tau, f["mdot_surf"], color=AMBER, lw=1.1, label="free surface")
    ax_m.plot(tau, f["mdot_wall"], color=SLATE, lw=1.1, label="wall (- = cond.)")
    ax_m.plot(tau, f["mdot_out"], color=RED, lw=1.2, label="ejected")
    ax_m.set_ylabel("mass flow (kg/s)", fontsize=8)
    ax_m.set_xlabel("time (s)", fontsize=8)
    ax_m.set_title("Mass flows", fontsize=9)
    ax_m.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)

    for ax in (ax_p, ax_pv, ax_T, ax_m):
        ax.tick_params(labelsize=7)
        ax.grid(color="#E2E8F0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    cursors = [ax.axvline(0, color=SLATE, lw=0.9) for ax in (ax_p, ax_T, ax_m)]
    (dot,) = ax_pv.plot([], [], "o", color=RED, ms=6, zorder=5)
    return fig, sch, cursors, dot


def main():
    args = parse_args()
    print("simulating ...", flush=True)
    p, f, m = prepare(args)
    print(m.summary(), flush=True)
    fig, sch, cursors, dot = build_figure(p, f, m)
    dt = float(f["tau"][1] - f["tau"][0])
    mf, mo = float(f["mdot_flash"].max()), float(f["mdot_out"].max())

    def frame(i):
        sch.update(f, i, dt, mf, mo)
        for c in cursors:
            c.set_xdata([f["tau"][i]] * 2)
        dot.set_data([f["V_s"][i]], [f["p_s"][i] / 1e3])
        return []

    anim = FuncAnimation(fig, frame, frames=len(f["tau"]), interval=1000 / args.fps, blit=False)
    if args.show:
        plt.show()
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = (PillowWriter(fps=args.fps) if args.out.suffix.lower() == ".gif" else
              FFMpegWriter(fps=args.fps, bitrate=2400, codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]))
    print(f"rendering {len(f['tau'])} frames -> {args.out}", flush=True)
    anim.save(args.out, writer=writer, dpi=args.dpi)
    print("done", flush=True)


if __name__ == "__main__":
    main()
