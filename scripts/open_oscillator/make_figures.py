"""Sensitivity figure of the open oscillator (run ``run_study.py`` first)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEAL, AMBER, GREY, SLATE = "#0F766E", "#B45309", "#94A3B8", "#334155"


def series(sw):
    rows = [r for r in sw["rows"] if "error" not in r]
    return ([r["value"] for r in rows], [r["steam_power"] / 1e3 for r in rows],
            [r["cop"] if r["m_out"] > 0 else 0.0 for r in rows])


def main():
    sw = json.loads((ROOT / "results" / "open_oscillator" / "sweeps.json").read_text())
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
    panels = [("h_cond", axs[0, 0], 300, "log"), ("D_h", axs[0, 1], 0.6, None),
              ("T_bulk", axs[1, 0], 60, None), ("P_assist", axs[1, 1], 20, None)]
    for name, ax, ref, scale in panels:
        x, P, cop = series(sw[name])
        ax.plot(x, P, "o-", color=TEAL, ms=4, label="h_cond 300" if name == "D_h" else None)
        ax.set_ylabel("net steam power (kW)", color=TEAL)
        ax2 = ax.twinx()
        ax2.plot(x, cop, "s--", color=AMBER, ms=3.5)
        ax2.axhline(1.0, color=AMBER, lw=0.6, ls=":")
        ax2.set_ylabel("COP", color=AMBER)
        ax2.spines["right"].set_visible(True)
        if name == "D_h":
            xw, Pw, copw = series(sw["D_h_wet"])
            ax.plot(xw, Pw, "o-", color=SLATE, ms=4, label="h_cond 2000 (wetter walls)")
            ax2.plot(xw, copw, "s--", color=GREY, ms=3.5)
            ax.legend(loc="upper left", frameon=False, fontsize=7)
        ax.set_xlabel(f"{sw[name]['label']} ({sw[name]['unit']})")
        ax.axvline(ref, color=GREY, lw=0.8, ls=":")
        ax.grid(color="#E2E8F0", lw=0.6)
        if scale:
            ax.set_xscale(scale)
    fig.suptitle("Open oscillator: one-at-a-time sensitivity (dotted: reference)", fontsize=9.5)
    fig.tight_layout()
    out = ROOT / "figures" / "open_oscillator" / "sensitivity.png"
    fig.savefig(out, dpi=150)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
