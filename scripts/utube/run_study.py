"""Run the reference case and the one-at-a-time sensitivity sweeps.

Results go to ``results/utube/`` as JSON so that the figures can be regenerated
without re-running the simulations:

    python scripts/utube/run_study.py                   # everything, ~4 min on one core
    python scripts/utube/run_study.py reference T_bulk  # only these parts (merged into sweeps.json)
    python scripts/utube/make_figures.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from liquid_piston.utube import (REFERENCE, balances, cycle_metrics,  # noqa: E402
                                 reference_drive, run_to_periodic)

OUT = ROOT / "results" / "utube"
N_CYCLES = 15

SWEEPS = {
    # name: (label, unit, values, function value -> (params, drive kwargs))
    "T_bulk": ("Bulk liquid temperature", "degC", [45, 50, 55, 60, 65, 70, 75],
               lambda v: (REFERENCE.with_(T_bulk=273.15 + v), {})),
    "stroke": ("Stroke amplitude", "m", [1.0, 1.1, 1.2, 1.3, 1.4, 1.45],
               lambda v: (REFERENCE, {"stroke": v})),
    "h_wall": ("Wall heat transfer coefficient", "W/m2K", [0, 10, 20, 50, 100],
               lambda v: (REFERENCE.with_(h_wall=float(v)), {})),
    "delta": ("Surface layer thickness", "mm", [0.3, 0.5, 1.0, 2.0],
              lambda v: (REFERENCE.with_(delta=v * 1e-3), {})),
    "h_bulk": ("Layer-to-bulk heat transfer coefficient", "W/m2K", [500, 1000, 2000, 4000, 8000],
               lambda v: (REFERENCE.with_(h_bulk=float(v)), {})),
    "sigma": ("Accommodation coefficient", "-", [0.01, 0.04, 0.2, 1.0],
              lambda v: (REFERENCE.with_(sigma=v), {})),
    "cav": ("Flash-boiling area factor", "-", [0, 1, 3, 10, 30],
            lambda v: (REFERENCE.with_(cav_area_factor=float(v)), {})),
    "freq": ("Drive frequency / U-tube frequency", "-", [0.6, 0.8, 1.0, 1.25, 1.5],
             lambda v: (REFERENCE, {"freq_ratio": v})),
}


def metrics_dict(m) -> dict:
    d = asdict(m)
    d.update(steam_power=m.steam_power, steam_rate=m.steam_rate,
             work_share=m.work_share, second_law_efficiency=m.second_law_efficiency)
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in d.items()}


def run_reference(t0: float) -> None:
    res = run_to_periodic(REFERENCE, reference_drive(), n_cycles=N_CYCLES)
    m = cycle_metrics(res)
    b = balances(res)
    ref = metrics_dict(m)
    ref.update(energy_error=b.energy_error, mass_error=b.mass_error,
               params=asdict(REFERENCE), drive=asdict(reference_drive()))
    (OUT / "reference.json").write_text(json.dumps(ref, indent=2))
    (OUT / "reference.txt").write_text(m.summary() + "\n")
    print(m.summary(), f"\n[{time.time() - t0:.1f} s]", flush=True)


def main(selected: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    if not selected:
        selected = ["reference", *SWEEPS]
    if "reference" in selected:
        run_reference(t0)
    selected = [s for s in selected if s in SWEEPS]
    path = OUT / "sweeps.json"
    sweeps = json.loads(path.read_text()) if path.exists() else {}
    for name in selected:
        label, unit, values, make = SWEEPS[name]
        rows = []
        for v in values:
            params, kw = make(v)
            mv = cycle_metrics(run_to_periodic(params, reference_drive(params, **kw),
                                               n_cycles=N_CYCLES))
            rows.append({"value": v, **metrics_dict(mv)})
            print(f"{name:7s} {v:>7} -> {mv.steam_power / 1e3:7.1f} kW, "
                  f"work share {mv.work_share:.3f}  [{time.time() - t0:.0f} s]", flush=True)
        sweeps[name] = {"label": label, "unit": unit, "rows": rows}
        path.write_text(json.dumps(sweeps, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
