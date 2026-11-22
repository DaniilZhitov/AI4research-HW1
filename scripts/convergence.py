"""Convergence of the eight central minibands with the number of plane-wave rings."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moire_dirac import (  # noqa: E402
    build_hamiltonian,
    central_bands,
    mini_bz_path,
)

L_NM = 10.0
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"


def path_bands(alpha: float, phi: float, rings: int) -> np.ndarray:
    """Return the central bands on the path in units of E_0."""
    scale = build_hamiltonian(0.0, rings=1, L=L_NM)
    bands = []
    for k in mini_bz_path(scale.g_norm):
        model = build_hamiltonian(alpha * scale.e0, rings=rings, k=k, phi=phi, L=L_NM)
        energies = np.linalg.eigvalsh(model.matrix) / model.e0
        bands.append(central_bands(energies, model.n_harmonics))
    return np.array(bands)


def error_between(alpha: float, phi: float, rings: int, reference: int) -> float:
    return float(
        np.abs(path_bands(alpha, phi, rings) - path_bands(alpha, phi, reference)).max()
    )


def plot_convergence(rows: list[dict], threshold: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    alphas = sorted({row["alpha"] for row in rows})
    for alpha, color in zip(alphas, SERIES_COLORS):
        series = [row for row in rows if row["alpha"] == alpha]
        rings = [row["rings"] for row in series]
        errors = [row["error"] for row in series]
        label = f"α = {alpha:g}".replace(".", ",")
        ax.semilogy(
            rings, errors, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=label,
        )
        ax.annotate(
            label, (rings[-1], errors[-1]), xytext=(8, 0),
            textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=9,
        )
    ax.axhline(threshold, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        "порог 10⁻³", (0.995, threshold), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", ha="right",
        color=TEXT_SECONDARY, fontsize=9,
    )
    ax.set_xlabel("Число колец N", color=TEXT_SECONDARY)
    ax.set_ylabel("Погрешность восьми центральных зон, E₀", color=TEXT_SECONDARY)
    ax.set_title(
        "Погрешность мини-зон падает ниже порога при малом числе колец",
        loc="left", color=TEXT_PRIMARY, fontsize=11,
    )
    ax.set_xticks(range(1, max(row["rings"] for row in rows) + 1))
    ax.set_xlim(0.8, max(row["rings"] for row in rows) + 0.7)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.tick_params(which="both", colors=TEXT_SECONDARY, length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="lower left", labelcolor=TEXT_PRIMARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def plot_bands(alpha: float, phi: float, rings: int, output: Path) -> None:
    scale = build_hamiltonian(0.0, rings=1, L=L_NM)
    path = mini_bz_path(scale.g_norm) / scale.g_norm
    distance = np.concatenate([[0.0], np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()])
    bands = path_bands(alpha, phi, rings)
    corners = [0.0, 1 / np.sqrt(3), 1 / np.sqrt(3) + 0.5 / np.sqrt(3), distance[-1]]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(distance, bands, color=SERIES_COLORS[0], linewidth=1.6)
    for corner in corners[1:-1]:
        ax.axvline(corner, color=GRID, linewidth=0.8)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set_xticks(corners, ["Γ", "K_m", "M_m", "Γ"])
    ax.set_xlim(distance[0], distance[-1])
    ax.set_ylabel("Энергия, E₀", color=TEXT_SECONDARY)
    alpha_text = f"{alpha:g}".replace(".", ",")
    ax.set_title(
        f"Диагностика, не финальный тест: α = {alpha_text}, φ = π/2, N = {rings}",
        loc="left", color=TEXT_PRIMARY, fontsize=11,
    )
    ax.tick_params(which="both", colors=TEXT_SECONDARY, length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.3])
    parser.add_argument("--phi", type=float, default=math.pi / 2)
    parser.add_argument("--max-rings", type=int, default=12)
    parser.add_argument("--stop-at", type=float, default=1e-3)
    parser.add_argument("--reference-offset", type=int, default=3)
    parser.add_argument("--control-rings", type=int, default=12)
    args = parser.parse_args()

    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    rows: list[dict] = []
    smallest: dict[float, int | None] = {}
    for alpha in args.alpha:
        smallest[alpha] = None
        for rings in range(1, args.max_rings + 1):
            reference = rings + args.reference_offset
            error = error_between(alpha, args.phi, rings, reference)
            below = error <= args.stop_at
            rows.append(
                {"alpha": alpha, "rings": rings, "reference_rings": reference,
                 "error": error, "below_threshold": below}
            )
            print(f"alpha={alpha:g} N={rings} ref={reference} error={error:.3e}", flush=True)
            if below:
                smallest[alpha] = rings
                break

    with (results / "convergence.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["alpha", "rings", "reference_rings", "error", "below_threshold"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "error": f"{row['error']:.3e}"})
    plot_convergence(rows, args.stop_at, results / "convergence.png")

    if any(value is None for value in smallest.values()):
        print("N_work is undefined")
        return 1

    working = max(smallest.values())
    print(f"smallest={smallest} N_work={working}", flush=True)

    with (results / "convergence_control.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "rings", "control_rings", "error"])
        for alpha in args.alpha:
            error = error_between(alpha, args.phi, working, args.control_rings)
            writer.writerow([alpha, working, args.control_rings, f"{error:.3e}"])
            print(f"control alpha={alpha:g} N={working} ref={args.control_rings} error={error:.3e}", flush=True)

    if 0.1 in args.alpha:
        plot_bands(0.1, args.phi, working, results / "bands_diag.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
