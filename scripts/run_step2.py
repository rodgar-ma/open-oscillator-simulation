"""Verificación visual del peldaño 2 — fricción cuadrática, envolvente hiperbólica."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utube.column import Column
from utube.dynamics import simulate, specific_energy

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

COL = Column()
X0 = 0.5
N_PERIODS = 10
T_END = N_PERIODS * 2 * np.pi / COL.omega_0
K = 4 * COL.loss_coefficient * COL.omega_0 / (3 * np.pi * COL.length)


def plot_envelope():
    sol = simulate(COL, X0, T_END, friction=True)
    t, x = sol.t, sol.y[0]

    peaks, _ = find_peaks(np.abs(x))
    hyperbolic = X0 / (1 + K * X0 * t)
    exponential = X0 * np.exp(-K * X0 * t)  # referencia: NO es el modelo, es cuadrático

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, x, lw=0.5, color="0.75", label="$z_L(t)$")
    ax.plot(t[peaks], np.abs(x[peaks]), "o", ms=3, label="envolvente numérica (picos)")
    ax.plot(t, hyperbolic, "r-", label=r"predicción hiperbólica $X_0/(1+kX_0t)$")
    ax.plot(t, exponential, "k--", lw=1, label="exponencial (referencia, no aplica)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$z_L$ [m]")
    ax.set_title(f"Peldaño 2 — decaimiento hiperbólico (k = {K:.5f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step2_envelope.png", dpi=150)
    plt.close(fig)


def plot_energy_dissipation():
    sol = simulate(COL, X0, T_END, friction=True)
    t, x, v = sol.t, sol.y[0], sol.y[1]
    e = specific_energy(x, v, COL)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t, e)
    ax1.set_ylabel("$e(t)$ [J/kg]")
    ax1.set_title("Peldaño 2 — disipación de energía por arrastre cuadrático")

    de_dt_numeric = np.gradient(e, t)
    de_dt_predicted = -COL.loss_coefficient * np.abs(v) ** 3
    ax2.plot(t, de_dt_numeric, label=r"$de/dt$ numérico ($\nabla e$)")
    ax2.plot(t, de_dt_predicted, "--", label=r"$-c\,|v|^3$ analítico")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("$de/dt$ [W/kg]")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "step2_energy_dissipation.png", dpi=150)
    plt.close(fig)


def main():
    plot_envelope()
    plot_energy_dissipation()
    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
