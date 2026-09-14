"""Verificación visual del peldaño 1 — U abierta, sin fricción."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utube.column import Column
from utube.dynamics import simulate, specific_energy

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

COL = Column()
X0 = 0.5
N_PERIODS = 10
T_END = N_PERIODS * 2 * np.pi / COL.omega_0


def plot_analytic_match():
    sol = simulate(COL, X0, T_END)
    exact = X0 * np.cos(COL.omega_0 * sol.t)
    error = np.abs(sol.y[0] - exact)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(sol.t, sol.y[0], label="numérico (RK45)")
    ax1.plot(sol.t, exact, "--", label=r"analítico $x_0\cos(\omega_0 t)$")
    ax1.set_ylabel("$z_L$ [m]")
    ax1.set_title(f"Peldaño 1 — U abierta sin fricción ({N_PERIODS} periodos)")
    ax1.legend()

    ax2.semilogy(sol.t, error, label="|numérico - analítico|")
    ax2.axhline(1e-8, color="r", ls=":", label="tolerancia 1e-8")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("error absoluto [m]")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "step1_analytic_match.png", dpi=150)
    plt.close(fig)


def plot_energy_conservation():
    sol = simulate(COL, X0, T_END)
    e = specific_energy(sol.y[0], sol.y[1], COL)
    rel_error = (e - e[0]) / e[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sol.t, rel_error)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$(e(t) - e_0)/e_0$")
    ax.set_title("Peldaño 1 — conservación de energía específica")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step1_energy_conservation.png", dpi=150)
    plt.close(fig)


def plot_isochronism():
    fig, ax = plt.subplots(figsize=(8, 4))
    for x0 in (0.1, 0.5, 1.0):
        sol = simulate(COL, x0, T_END)
        ax.plot(sol.t, sol.y[0] / x0, label=f"$x_0$ = {x0} m")
    ax.set_xlim(0, 3 * 2 * np.pi / COL.omega_0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$z_L / x_0$ (normalizado)")
    ax.set_title("Peldaño 1 — isocronía: mismo periodo para toda amplitud")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step1_isochronism.png", dpi=150)
    plt.close(fig)


def main():
    plot_analytic_match()
    plot_energy_conservation()
    plot_isochronism()
    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
