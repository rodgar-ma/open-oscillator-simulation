"""Reference case and sensitivity sweeps for the open oscillator.

    python scripts/run_open_study.py                    # everything, ~5 min
    python scripts/run_open_study.py reference h_cond   # selected parts
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liquid_piston.open_oscillator import (OPEN_REFERENCE, open_cycle_metrics,  # noqa: E402
                                           open_energy_error, simulate_open)

OUT = ROOT / "results"
T_END = 60.0


def scaled(D):
    """Geometrically similar unit with high-pipe diameter D (same height)."""
    s = (D / OPEN_REFERENCE.D_h) ** 2
    ref = OPEN_REFERENCE
    return ref.with_(D_h=D, D_t=ref.D_t * D / ref.D_h, V_g0=ref.V_g0 * s,
                     valve_area=ref.valve_area * s, V_dead=ref.V_dead * s)


SWEEPS = {
    "h_cond": ("Wall condensation coefficient", "W/m2K", [30, 100, 300, 1000, 3000],
               lambda v: OPEN_REFERENCE.with_(h_cond=float(v))),
    "D_h": ("High-pipe diameter (similar geometry)", "m", [0.3, 0.45, 0.6, 0.8, 1.0], scaled),
    "D_h_wet": ("High-pipe diameter, h_cond = 2000 W/m2K", "m", [0.3, 0.45, 0.6, 0.8, 1.0, 1.3],
                lambda v: scaled(v).with_(h_cond=2000.0)),
    "T_bulk": ("Waste-heat (bulk) temperature", "degC", [45, 50, 55, 60, 65],
               lambda v: OPEN_REFERENCE.with_(T_bulk=273.15 + v)),
    "P_assist": ("Drive assist pressure", "kPa", [12, 16, 20, 25, 30],
                 lambda v: OPEN_REFERENCE.with_(P_assist=v * 1e3)),
}


def row(m) -> dict:
    d = asdict(m)
    d.update(cop=m.cop, steam_power=m.steam_power, compression_ratio=m.compression_ratio,
             second_law_efficiency=m.second_law_efficiency)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def main(selected):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    selected = selected or ["reference", *SWEEPS]
    if "reference" in selected:
        res = simulate_open(OPEN_REFERENCE, T_END)
        m, prev = open_cycle_metrics(res), open_cycle_metrics(res, which=-2)
        ref = row(m)
        ref.update(energy_error=open_energy_error(res),
                   periodicity=abs(m.m_out - prev.m_out) / max(m.m_out, 1e-12),
                   params=asdict(OPEN_REFERENCE))
        (OUT / "open_reference.json").write_text(json.dumps(ref, indent=2))
        (OUT / "open_reference.txt").write_text(m.summary() + "\n")
        print(m.summary(), f"\nenergy error {ref['energy_error']:.1e}, "
              f"cycle-to-cycle {ref['periodicity']:.1e} [{time.time() - t0:.0f} s]", flush=True)
    path = OUT / "open_sweeps.json"
    sweeps = json.loads(path.read_text()) if path.exists() else {}
    for name in [s for s in selected if s in SWEEPS]:
        label, unit, values, make = SWEEPS[name]
        rows = []
        for v in values:
            try:
                m = open_cycle_metrics(simulate_open(make(v), T_END))
                rows.append({"value": v, **row(m)})
                print(f"{name:8s} {v:>6} -> {m.steam_power / 1e3:6.1f} kW, COP {m.cop:5.2f}, "
                      f"ratio {m.compression_ratio:4.1f}  [{time.time() - t0:.0f} s]", flush=True)
            except RuntimeError as e:  # e.g. the slug hits the top
                rows.append({"value": v, "error": str(e)})
                print(f"{name:8s} {v:>6} -> {e}", flush=True)
        sweeps[name] = {"label": label, "unit": unit, "rows": rows}
        path.write_text(json.dumps(sweeps, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
