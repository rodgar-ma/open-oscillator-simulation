# liquid-piston-0d

A zero-dimensional model of a liquid piston oscillating in a U-tube with a
vapour chamber: the column falls and the chamber flashes, the column rises and
compresses the vapour, and a check valve ejects steam above the outlet pressure.

I wrote it to understand the physics behind open oscillatory waste-heat-to-steam
systems such as the one Hydram Research has described publicly. It is a
learning and exploration tool built only from public information. The geometry
and parameters are illustrative and are **not** Hydram's design, and the
results have not been validated against measurements.

![Animated simulation of the reference case](figures/evaporator.gif)

## What the model contains

| Sub-model | Approach | Switch |
| --- | --- | --- |
| Column dynamics | Rigid U-tube column, hydrostatic restoring force, quadratic friction | `K_loss` |
| Drive | Prescribed pressure on leg 2, or prescribed motion with the required drive pressure computed | `PressureDrive` / `KinematicDrive` |
| Vapour | Uniform ideal gas, open-system energy balance | — |
| Saturation properties | IAPWS-IF97 region 4 (checked against the standard's verification values and CoolProp) | — |
| Free-surface phase change | Hertz-Knudsen-Schrage | `sigma`, `area_factor` |
| Surface layer | Thin liquid layer with its own temperature, coupled to the bulk | `delta`, `h_bulk` |
| Flash boiling (cavitation) | One-way HKS evaporation from bubble interfaces while the chamber is below p_sat(T_bulk) | `cav_area_factor` |
| Fog | Relaxation of supersaturated vapour back to saturation; droplets rain out | `fog_tau` |
| Chamber walls | Sensible heat exchange with walls at bulk temperature | `h_wall` |
| Ejection valve | Isentropic orifice flow, subsonic or choked, no reverse flow | `valve_area`, `valve_cd`, `p_out` |

Liquid and vapour enthalpies share a single reference, chosen so the latent
heat is exact at 100 °C and within 1 % of steam tables from 7 to 127 °C. That
is what lets the energy balances close to integrator precision.

## Quick start

```bash
conda env create -f environment.yml   # creates the "hydram" environment, Python 3.12
conda activate hydram
pytest                                # 42 tests, ~65 s
python scripts/run_study.py           # reference case + sensitivity sweeps, ~4 min
python scripts/make_figures.py
python scripts/animate_evaporator.py  # animation -> figures/evaporator.mp4
```

The animation script takes the main operating parameters on the command line,
for example `--t-bulk 70 --stroke 1.4 --h-wall 50`, and `--show` opens an
interactive window instead of saving a file.

```python
from liquid_piston import REFERENCE, reference_drive, run_to_periodic, cycle_metrics

res = run_to_periodic(REFERENCE, reference_drive(), n_cycles=15)
print(cycle_metrics(res).summary())
```

## Verification ladder

Each rung switches on one more piece of physics and is checked against
something that does not come from the model itself.

| Rung | Physics | Check | Test file |
| --- | --- | --- | --- |
| 0 | Properties, valve | IAPWS-IF97 verification values, CoolProp, choked/incompressible limits | `test_properties.py`, `test_valve.py` |
| 1 | Frictionless U-tube | Period 2π√(L/2g) to 1e-5; energy conserved to 1e-6 | `test_rung1_2_oscillator.py` |
| 2 | Quadratic friction | Amplitude decay against the energy-method envelope (3 %) | `test_rung1_2_oscillator.py` |
| 3 | Gas spring | p V^γ constant to 1e-6; small-amplitude frequency to 1e-4 | `test_rung3_gas_spring.py` |
| 4 | Phase change | Still chamber relaxes to p_sat(T_bulk); evaporative cooling of the surface | `test_rung4_5_phase_change.py` |
| 5 | Surface layer, flash, fog | Mass and energy balances close | `test_rung4_5_phase_change.py` |
| 6 | Full cycle | First law per cycle (~1e-5), second law (exergy delivered ≤ drive work), solver-tolerance convergence | `test_rung6_cycle.py` |

The energy check in `analysis.py` is written independently of the right-hand
side: stored energy comes from the states and boundary fluxes from their own
formulas, so a sign or reference error shows up as a closure error.

![Verification](figures/verification.png)

## Reference case

Water at 60 °C, D = 1 m, L = 10 m (about 7.7 t), headspace swept from 2.8 m to
0.2 m at the natural frequency of the plain U-tube (period 4.5 s), outlet at
155 kPa.

| Quantity | Value |
| --- | --- |
| Steam ejected | 99 g per cycle (79 kg/h) |
| Net steam power (relative to make-up water) | 62 kW |
| Drive work per cycle | 69 kJ (60 kJ compression, 9 kJ friction) |
| Waste heat per cycle | 207 kJ |
| Share of steam energy supplied as work | 25 % |
| Second-law efficiency (exergy to the stream / drive work) | 63 % |
| Peak chamber pressure | 185 kPa |
| Ejected steam temperature | 297 °C (adiabatic walls) |

![p-V diagram](figures/pv.png)

![One cycle of the reference case](figures/cycle.png)

## What the parameter study shows

![Sensitivity](figures/sensitivity.png)

1. **Free-surface evaporation alone cannot refill the chamber.** With
   `cav_area_factor = 0` the valve never opens. Bulk flash boiling during the
   expansion stroke is what fills the chamber, and the result saturates once
   the bubble interface is a few times the pipe cross-section.
2. **The main loss is condensation back onto the free surface during
   compression**, and it is controlled by how fast the surface heats up, not by
   interface kinetics. A thinner or more weakly coupled surface layer heats
   faster and limits condensation. The accommodation coefficient barely matters
   between 0.01 and 1.
3. **There are clear thresholds.** At this stroke, no steam is ejected with
   the bulk at 50 °C or below (the threshold lies between 50 and 55 °C); at
   60 °C, none with a stroke of 1.1 m or less (threshold between 1.1 and 1.2 m).
   Above the thresholds, steam output grows quickly and the work share falls.
4. **Walls set the superheat.** With adiabatic walls the ejected steam is very
   superheated (~300 °C). Hydram has reported steam at 122 °C and 155 kPa,
   which suggests strong heat exchange with the walls. Wall heat transfer is
   the parameter that most changes the output temperature here, and wall
   condensation, which this model leaves out, would matter even more.

## Open oscillator (high pipe + side tank)

`open_oscillator.py` models the architecture Hydram Dynamics describes on its
website (https://hydram.is/technical): a liquid slug oscillates between a tall
high pipe, whose top holds the steam, and a wide side tank with a gas cushion,
and a small assist pressure in the direction of travel sustains the
oscillation. It follows the modelling route suggested there: slug kinematics,
the side-tank gas bubble and the steam in the high pipe, coupled and stepped in
time. Geometry and parameters remain illustrative.

![Open oscillator animation](figures/open_oscillator.gif)

What it adds over the U-tube model:

- **Energy-based slug dynamics.** The inertia M(z) changes as liquid moves
  between sections of different area; the Lagrangian term 1/2 M'(z) u^2 keeps
  energy conserved (checked to 1e-9 in the conservative limit).
- **Direction-dependent losses:** entrance loss into the high pipe, Borda-Carnot
  exit loss into the tank.
- **Gas cushion** in the side tank (adiabatic ideal gas).
- **Wet-wall phase change** with an effective coefficient `h_cond` and a heated
  wall node, the loss Hydram says forces large diameters.
- **Feedback drive** `P_assist * tanh(u / u_ref)`, so the stroke and period come
  out of the dynamics instead of being prescribed.

Reference case (0.6 m x 9 m pipe, water at 60 °C, outlet 125 kPa): a 2.5 s
limit cycle, compression ratio 7.2, 36 g of steam per cycle (39 kW net) and
COP 1.9. Hydraulic losses take 60 % of the drive work.

![Open oscillator sensitivity](figures/open_sensitivity.png)

- **Wall condensation is the dominant uncertainty.** Output falls steeply with
  `h_cond` and stops between 1,000 and 3,000 W/m²K. Clean film condensation on
  bare steel would be higher still, so the real effective value is something
  to calibrate against measurements.
- **Bigger pipes work better.** With strongly wetted walls a 0.3 m pipe never
  ejects and COP only passes 1 near 1 m, which agrees qualitatively with
  Hydram's remark that the system needs to be large.
- Warmer waste heat and a stronger assist both raise output and COP.

```bash
python scripts/run_open_study.py         # reference case and sweeps, ~5 min
python scripts/make_open_figures.py
python scripts/animate_open_oscillator.py --h-cond 2000   # try wetter walls
```

## Limitations

- Lumped (0D): no pressure waves, sloshing, stratification or spatial
  temperature fields in the vapour.
- Flash boiling, the surface layer and fog are effective sub-models whose
  parameters (`cav_area_factor`, `delta`, `h_bulk`, `fog_tau`) would have to
  be fitted to experiments.
- Walls exchange sensible heat only. Condensation on walls below the
  saturation temperature is not modelled.
- No non-condensable gas. Even a small amount of air would change both
  evaporation and condensation strongly.
- Ideal-gas vapour (within 2 % of real steam density at the operating points).
- The kinematic drive prescribes the motion. The coupled dynamic drive is
  implemented and verified for rungs 1–3 but not used for the cycle study.

## Next steps

1. Wall condensation with a wall-temperature node.
2. Non-condensable gas in the chamber.
3. Coupled resonant drive: find the forcing that sustains the cycle instead of
   prescribing the motion.
4. Calibrate the effective parameters against measured pressure and
   temperature traces.

## Layout

```
src/liquid_piston/
    properties.py   IAPWS-IF97 saturation + consistent caloric model
    valve.py        compressible orifice flow
    model.py        parameters, drives, right-hand side, simulate()
    analysis.py     energy/mass balances, per-cycle metrics, second law
    cases.py        reference case
    open_oscillator.py  high pipe + side tank model, metrics and energy check
scripts/
    run_study.py    reference case and sensitivity sweeps -> results/
    make_figures.py figures/ from results/
    animate_evaporator.py  animated U-tube schematic with pressure, p-V, temperatures, flows
    run_open_study.py, make_open_figures.py, animate_open_oscillator.py  same for the open oscillator
tests/              verification ladder (pytest)
```

Author: Miguel Ángel Rodríguez García
