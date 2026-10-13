"""Animated simulation of the inertial evaporator (liquid piston in a U-tube).

Runs the 0D model to a periodic state and animates the last cycles:

* left: schematic of the U-tube. Liquid level, vapour coloured by its
  temperature, surface layer coloured by its temperature, flash-boiling
  bubbles in leg 1 and steam puffs from the valve when it is open;
* right: chamber pressure, p-V diagram, temperatures and mass flows, with a
  cursor that follows the animation.

Examples
--------
    python scripts/animate_evaporator.py                       # MP4 of the reference case
    python scripts/animate_evaporator.py --out evaporator.gif  # GIF instead
    python scripts/animate_evaporator.py --t-bulk 70 --stroke 1.4
    python scripts/animate_evaporator.py --show                # interactive window
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liquid_piston import REFERENCE, cycle_metrics, reference_drive, run_to_periodic  # noqa: E402
from liquid_piston import properties as wp  # noqa: E402

TEAL, AMBER, SLATE, RED, GREY = "#0F766E", "#B45309", "#334155", "#B91C1C", "#94A3B8"
WATER = "#2563EB"
T_MIN_C, T_MAX_C = 40.0, 320.0  # colour scale for temperatures, degC


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-bulk", type=float, default=REFERENCE.T_bulk - 273.15, help="bulk liquid temperature, degC")
    ap.add_argument("--stroke", type=float, default=1.3, help="stroke amplitude, m")
    ap.add_argument("--h-wall", type=float, default=REFERENCE.h_wall, help="wall heat transfer coefficient, W/m2K")
    ap.add_argument("--warmup", type=int, default=15, help="cycles to reach the periodic state")
    ap.add_argument("--cycles", type=int, default=2, help="cycles to animate")
    ap.add_argument("--frames-per-cycle", type=int, default=110)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "evaporator.mp4",
                    help="output file (.mp4 or .gif)")
    ap.add_argument("--show", action="store_true", help="open an interactive window instead of saving")
    return ap.parse_args()


def simulate(args):
    params = REFERENCE.with_(T_bulk=273.15 + args.t_bulk, h_wall=args.h_wall)
    drive = reference_drive(params, stroke=args.stroke)
    if drive.x_center + drive.stroke >= params.H0:
        raise SystemExit(f"stroke too large: the liquid would hit the chamber top (H0 = {params.H0} m)")
    res = run_to_periodic(params, drive, n_cycles=args.warmup + args.cycles)
    metrics = cycle_metrics(res)
    T = drive.period
    t = np.linspace(res.t[-1] - args.cycles * T, res.t[-1], args.cycles * args.frames_per_cycle)
    fx = res.flux_series(t)
    fx["tau"] = (t - t[0]) / T  # time in periods
    return params, drive, fx, metrics


def phase_label(f, i):
    if f["mdot_out"][i] > 0:
        return "Ejection: valve open", RED
    if f["v"][i] > 0:
        return "Compression", AMBER
    if f["mdot_flash"][i] > 0:
        return "Expansion: flash boiling", TEAL
    return "Re-expansion", SLATE


class Schematic:
    """U-tube drawing in metres (not to scale horizontally)."""

    def __init__(self, ax, params, cmap, norm):
        self.ax, self.p, self.cmap, self.norm = ax, params, cmap, norm
        D = params.D
        self.x1 = (-D / 2, D / 2)              # leg 1 (chamber side)
        self.x2 = (2.0 * D, 3.0 * D)           # leg 2 (drive side)
        self.y_bot = -D                        # bottom of the horizontal pipe
        self.h_eq = 0.5 * (params.L - 2.5 * D)  # equilibrium level so the column length is ~L
        self.y_top1 = self.h_eq + params.H0    # chamber lid
        self.y_top2 = self.y_top1 + 0.8
        self.valve = (0.0, self.y_top1 + 0.35)
        self.outlet = (0.0, self.y_top1 + 0.9)

        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 4.4 * D)
        ax.set_ylim(self.y_bot - 0.4, self.y_top1 + 3.4)
        ax.axis("off")

        # liquid (bottom pipe + two legs) and vapour
        self.bottom = Rectangle((self.x1[0], self.y_bot), self.x2[1] - self.x1[0], -self.y_bot,
                                color=WATER, alpha=0.85, lw=0)
        self.liq1 = Rectangle((self.x1[0], 0), D, 1, color=WATER, alpha=0.85, lw=0)
        self.liq2 = Rectangle((self.x2[0], 0), D, 1, color=WATER, alpha=0.85, lw=0)
        self.vap = Rectangle((self.x1[0], 0), D, 1, lw=0)
        self.layer = Rectangle((self.x1[0], 0), D, 0.12, lw=0)
        for patch in (self.bottom, self.liq1, self.liq2, self.vap, self.layer):
            ax.add_patch(patch)

        # tube walls
        wall = dict(color=SLATE, lw=2.2, solid_capstyle="round")
        ax.plot([self.x1[0]] * 2, [self.y_bot, self.y_top1], **wall)
        ax.plot([self.x1[1]] * 2, [0, self.y_top1], **wall)
        ax.plot([self.x1[0], self.x1[1]], [self.y_top1] * 2, **wall)  # chamber lid
        ax.plot([self.x2[0]] * 2, [0, self.y_top2], **wall)
        ax.plot([self.x2[1]] * 2, [self.y_bot, self.y_top2], **wall)
        ax.plot([self.x1[1], self.x2[0]], [0, 0], **wall)
        ax.plot([self.x1[0], self.x2[1]], [self.y_bot] * 2, **wall)
        ax.plot([self.x2[0] - 0.15, self.x2[1] + 0.15], [self.h_eq] * 2,
                color=GREY, lw=0.8, ls=":")
        ax.plot([self.x1[0] - 0.15, self.x1[1] + 0.15], [self.h_eq] * 2, color=GREY, lw=0.8, ls=":")

        # valve and outlet pipe
        ax.plot([0, 0], [self.y_top1, self.outlet[1]], color=SLATE, lw=2.2)
        vx, vy = self.valve
        self.valve_patch = Polygon([[vx - 0.22, vy - 0.15], [vx + 0.22, vy + 0.15],
                                    [vx + 0.22, vy - 0.15], [vx - 0.22, vy + 0.15]],
                                   closed=True, fc="white", ec=SLATE, lw=1.5, zorder=5)
        ax.add_patch(self.valve_patch)
        ax.text(vx + 0.35, vy, "check valve", fontsize=7, color=SLATE, va="center")

        # labels
        ax.text(np.mean(self.x1), self.y_bot - 0.3, "leg 1\nvapour chamber", ha="center", va="top", fontsize=7)
        ax.text(np.mean(self.x2), self.y_bot - 0.3, "leg 2\ndrive", ha="center", va="top", fontsize=7)
        self.drive_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                       arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
        self.p2_text = ax.text(np.mean(self.x2), self.y_top2 + 0.15, "", ha="center", fontsize=7.5)
        self.chamber_text = ax.text(1.0, self.y_top1 + 2.75, "", fontsize=8, va="top", family="monospace",
                                    bbox=dict(fc="white", ec="#CBD5E1", boxstyle="round,pad=0.4"))
        self.phase_text = ax.text(-0.95, self.y_top1 + 3.3, "", fontsize=11, weight="bold", va="top")

        # particles
        self.rng = np.random.default_rng(1)
        self.bubbles = np.empty((0, 3))  # x, y, age
        self.steam = np.empty((0, 3))    # x, y, age
        self.bubble_sc = ax.scatter([], [], s=10, facecolors="white", edgecolors="#BFDBFE",
                                    linewidths=0.6, alpha=0.9, zorder=4)
        self.steam_sc = ax.scatter([], [], s=60, color="#E2E8F0", edgecolors="none", zorder=6)

    def update(self, f, i, dt_frame, mdot_flash_max, mdot_out_max):
        p, D = self.p, self.p.D
        x = f["x"][i]
        h1, h2 = self.h_eq + x, self.h_eq - x
        self.liq1.set_height(h1)
        self.liq2.set_height(h2)
        self.vap.set_y(h1)
        self.vap.set_height(self.y_top1 - h1)
        self.vap.set_color(self.cmap(self.norm(f["T_v"][i] - 273.15)))
        self.layer.set_y(h1 - 0.12)
        self.layer.set_color(self.cmap(self.norm(f["T_s"][i] - 273.15)))

        # leg 2 surface moves opposite to leg 1
        vx = np.mean(self.x2)
        self.drive_arrow.xy = (vx, h2 - 0.6 * np.sign(f["v"][i]) * min(1.0, abs(f["v"][i])))
        self.drive_arrow.set_position((vx, h2))
        self.p2_text.set_text(f"p2 = {f['p2'][i] / 1e3:5.0f} kPa")

        open_ = f["mdot_out"][i] > 0
        self.valve_patch.set_facecolor(RED if open_ else "white")

        # flash-boiling bubbles in leg 1 rise and pop at the surface
        n_new = self.rng.poisson(14 * f["mdot_flash"][i] / mdot_flash_max) if mdot_flash_max > 0 else 0
        if n_new:
            new = np.column_stack([self.rng.uniform(self.x1[0] + 0.08, self.x1[1] - 0.08, n_new),
                                   self.rng.uniform(0.0, max(h1 - 0.2, 0.1), n_new),
                                   np.zeros(n_new)])
            self.bubbles = np.vstack([self.bubbles, new])
        if len(self.bubbles):
            self.bubbles[:, 1] += 1.6 * dt_frame
            self.bubbles[:, 2] += dt_frame
            self.bubbles = self.bubbles[self.bubbles[:, 1] < h1 - 0.05]
        self.bubble_sc.set_offsets(self.bubbles[:, :2] if len(self.bubbles) else np.empty((0, 2)))

        # steam puffs from the outlet
        n_new = self.rng.poisson(8 * f["mdot_out"][i] / mdot_out_max) if mdot_out_max > 0 else 0
        if n_new:
            ox, oy = self.outlet
            new = np.column_stack([ox + self.rng.normal(0, 0.05, n_new),
                                   oy + self.rng.uniform(0, 0.1, n_new), np.zeros(n_new)])
            self.steam = np.vstack([self.steam, new])
        if len(self.steam):
            self.steam[:, 0] += self.rng.normal(0.25, 0.25, len(self.steam)) * dt_frame
            self.steam[:, 1] += 1.8 * dt_frame
            self.steam[:, 2] += dt_frame
            self.steam = self.steam[self.steam[:, 2] < 1.4]
        if len(self.steam):
            self.steam_sc.set_offsets(self.steam[:, :2])
            alpha = np.clip(1.0 - self.steam[:, 2] / 1.4, 0, 1)
            self.steam_sc.set_alpha(None)
            self.steam_sc.set_facecolors([(0.85, 0.88, 0.92, a) for a in alpha])
            self.steam_sc.set_sizes(40 + 160 * self.steam[:, 2])
        else:
            self.steam_sc.set_offsets(np.empty((0, 2)))

        label, color = phase_label(f, i)
        self.phase_text.set_text(label)
        self.phase_text.set_color(color)
        T_sat = wp.T_sat(f["p_v"][i]) - 273.15
        self.chamber_text.set_text(
            f"chamber   p     {f['p_v'][i] / 1e3:6.1f} kPa\n"
            f"vapour    T_v   {f['T_v'][i] - 273.15:6.1f} degC\n"
            f"          T_sat {T_sat:6.1f} degC\n"
            f"surface   T_s   {f['T_s'][i] - 273.15:6.1f} degC")


def build_figure(params, drive, f, metrics):
    cmap = plt.get_cmap("plasma")
    norm = Normalize(T_MIN_C, T_MAX_C)
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.45, wspace=0.36,
                          left=0.01, right=0.98, top=0.88, bottom=0.09)
    ax_s = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_pv = fig.add_subplot(gs[0, 2])
    ax_T = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_m = fig.add_subplot(gs[1, 2])

    fig.suptitle(
        f"Inertial evaporator, 0D model  |  water {params.T_bulk - 273.15:.0f} degC, "
        f"stroke {drive.stroke:.2f} m, period {drive.period:.1f} s, outlet {params.p_out / 1e3:.0f} kPa\n"
        f"per cycle: {metrics.m_out * 1e3:.0f} g steam, {metrics.steam_power / 1e3:.0f} kW net, "
        f"work share {metrics.work_share * 100 if metrics.m_out > 0 else float('nan'):.0f} %",
        fontsize=10.5)
    sch = Schematic(ax_s, params, cmap, norm)
    cax = ax_s.inset_axes([0.86, 0.22, 0.035, 0.38])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("vapour / surface\ntemperature (degC)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    tau = f["tau"]
    ax_p.plot(tau, f["p_v"] / 1e3, color=TEAL, lw=1.4)
    ax_p.axhline(params.p_out / 1e3, color=AMBER, ls="--", lw=1)
    ax_p.axhline(wp.p_sat(params.T_bulk) / 1e3, color=GREY, ls=":", lw=1)
    ax_p.text(0.01, params.p_out / 1e3 + 4, "outlet", color=AMBER, fontsize=7)
    ax_p.set_ylabel("chamber pressure (kPa)", fontsize=8)
    ax_p.set_title("Pressure", fontsize=9)

    ax_T.plot(tau, f["T_v"] - 273.15, color=TEAL, lw=1.4, label="vapour")
    ax_T.plot(tau, f["T_s"] - 273.15, color=AMBER, lw=1.4, label="surface")
    ax_T.plot(tau, wp.T_sat(f["p_v"]) - 273.15, color=GREY, ls=":", lw=1, label="T_sat(p)")
    ax_T.set_ylabel("temperature (degC)", fontsize=8)
    ax_T.set_xlabel("time (periods)", fontsize=8)
    ax_T.legend(fontsize=7, loc="upper right", frameon=False)
    ax_T.set_title("Temperatures", fontsize=9)

    ax_pv.loglog(f["V"], f["p_v"] / 1e3, color=TEAL, lw=1.2)
    ax_pv.axhline(params.p_out / 1e3, color=AMBER, ls="--", lw=1)
    ax_pv.set_xlabel("vapour volume (m$^3$)", fontsize=8)
    ax_pv.set_ylabel("pressure (kPa)", fontsize=8)
    ax_pv.set_title("p-V diagram", fontsize=9)
    ax_pv.minorticks_off()
    ax_pv.set_yticks([20, 50, 100, 200], ["20", "50", "100", "200"])
    ax_pv.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])

    ax_m.plot(tau, f["mdot_flash"], color=TEAL, lw=1.2, label="flash boiling")
    ax_m.plot(tau, f["mdot_evap"], color=AMBER, lw=1.2, label="surface (+evap/-cond)")
    ax_m.plot(tau, f["mdot_out"], color=RED, lw=1.2, label="ejected")
    ax_m.set_ylabel("mass flow (kg/s)", fontsize=8)
    ax_m.set_xlabel("time (periods)", fontsize=8)
    ax_m.legend(fontsize=7, loc="upper right", frameon=False)
    ax_m.set_title("Mass flows", fontsize=9)

    for ax in (ax_p, ax_T, ax_pv, ax_m):
        ax.tick_params(labelsize=7)
        ax.grid(color="#E2E8F0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    cursors = [ax.axvline(0, color=SLATE, lw=0.9) for ax in (ax_p, ax_T, ax_m)]
    (pv_dot,) = ax_pv.plot([], [], "o", color=RED, ms=6, zorder=5)
    return fig, sch, cursors, pv_dot


def main():
    args = parse_args()
    print("simulating ...", flush=True)
    params, drive, f, metrics = simulate(args)
    print(metrics.summary(), flush=True)
    fig, sch, cursors, pv_dot = build_figure(params, drive, f, metrics)

    dt_frame = drive.period / args.frames_per_cycle
    mf_max, mo_max = float(f["mdot_flash"].max()), float(f["mdot_out"].max())

    def frame(i):
        sch.update(f, i, dt_frame, mf_max, mo_max)
        for c in cursors:
            c.set_xdata([f["tau"][i]] * 2)
        pv_dot.set_data([f["V"][i]], [f["p_v"][i] / 1e3])
        return []

    anim = FuncAnimation(fig, frame, frames=len(f["tau"]), interval=1000 / args.fps, blit=False)
    if args.show:
        plt.show()
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=2400, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    print(f"rendering {len(f['tau'])} frames -> {args.out}", flush=True)
    anim.save(args.out, writer=writer, dpi=args.dpi)
    print("done", flush=True)


if __name__ == "__main__":
    main()
