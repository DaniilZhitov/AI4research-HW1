"""The eight correctness checks of the plane-wave Hamiltonian.

Deviations are in units of E_0 = hbar v_F |G|, except the weak-potential check,
which reports a relative deviation of the velocity coefficient from 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian import (
    build_hamiltonian,
    central_bands,
    mini_bz_path,
    number_of_harmonics,
    sample_mini_bz,
)

L_NM = 10.0
SEED = 0
RINGS = (3, 8)
ALPHAS = (0.05, 0.1, 0.3)
PHI = np.pi / 2
N_K = 20
N_R = 40

_REFERENCE = build_hamiltonian(0.0, rings=1, L=L_NM)
E0 = _REFERENCE.e0
G_NORM = _REFERENCE.g_norm


@dataclass(frozen=True)
class CheckResult:
    name: str
    tolerance: float
    max_deviation: float
    passed: bool
    control_deviation: float | None = None
    control_threshold: float | None = None


def _h(alpha: float, rings: int, k, phi: float = PHI, valley: int = 1) -> np.ndarray:
    return build_hamiltonian(
        alpha * E0, rings=rings, k=k, phi=phi, L=L_NM, valley=valley
    ).matrix


def _spectrum(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(matrix) / E0


def _k_points(n_points: int = N_K) -> np.ndarray:
    return sample_mini_bz(G_NORM, n_points, np.random.default_rng(SEED))


def _shell_vectors() -> np.ndarray:
    angles = np.arange(6) * np.pi / 3
    return G_NORM * np.column_stack([np.cos(angles), np.sin(angles)])


def _harmonic_vectors(index: np.ndarray) -> np.ndarray:
    b1 = G_NORM * np.array([1.0, 0.0])
    b2 = G_NORM * np.array([-0.5, np.sqrt(3) / 2])
    a_rows = index[index["sublattice"] == "A"]
    return a_rows["m"][:, None] * b1 + a_rows["n"][:, None] * b2


def _potential(r: np.ndarray, amplitude: float, phi: float) -> np.ndarray:
    shell = _shell_vectors()[[0, 2, 4]]
    return 2 * amplitude * np.cos(r @ shell.T + phi).sum(axis=-1)


def check_hermiticity() -> CheckResult:
    tolerance = 1e-12
    rng = np.random.default_rng(SEED)
    random_phases = rng.uniform(0, 2 * np.pi, size=N_K)
    worst = 0.0
    for rings in RINGS:
        for alpha in ALPHAS:
            for k, random_phi in zip(_k_points(), random_phases):
                for phi in (PHI, random_phi):
                    for valley in (1, -1):
                        h = _h(alpha, rings, k, phi, valley)
                        worst = max(worst, np.abs(h - h.conj().T).max() / E0)
    return CheckResult("Эрмитовость", tolerance, worst, worst <= tolerance)


def check_free_limit() -> CheckResult:
    tolerance = 1e-10
    worst = 0.0
    for rings in RINGS:
        for k in mini_bz_path(G_NORM):
            model = build_hamiltonian(0.0, rings=rings, k=k, L=L_NM)
            numeric = central_bands(_spectrum(model.matrix), model.n_harmonics)
            q = np.linalg.norm(k + _harmonic_vectors(model.index), axis=1)
            exact = np.sort(np.concatenate([-q, q])) * model.hbar_vf / E0
            worst = max(
                worst,
                np.abs(numeric - central_bands(exact, model.n_harmonics)).max(),
            )
    return CheckResult("Свободный предел", tolerance, worst, worst <= tolerance)


def check_coupling_structure() -> CheckResult:
    tolerance = 1e-10
    shell = _shell_vectors()
    worst = 0.0
    for rings in RINGS:
        for alpha in ALPHAS:
            for phi in (PHI, 0.7):
                model = build_hamiltonian(
                    alpha * E0,
                    rings=rings,
                    k=_k_points(1)[0],
                    phi=phi,
                    L=L_NM,
                    as_tensor=True,
                )
                g = _harmonic_vectors(model.index)
                difference = g[:, None, :] - g[None, :, :]
                expected = np.zeros((len(g), len(g)), dtype=complex)
                for j, vector in enumerate(shell):
                    mask = np.linalg.norm(difference - vector, axis=2) < 1e-9 * G_NORM
                    phase = phi if j % 2 == 0 else -phi
                    expected[mask] = alpha * E0 * np.exp(1j * phase)

                actual = model.tensor.transpose(0, 2, 1, 3)
                target = np.einsum("ij,st->ijst", expected, np.eye(2))
                off_diagonal = ~np.eye(len(g), dtype=bool)
                worst = max(
                    worst, np.abs(actual - target)[off_diagonal].max() / E0
                )
    return CheckResult("Структура связи", tolerance, worst, worst <= tolerance)


def check_rotation() -> CheckResult:
    tolerance = 1e-10
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    rotation = np.array([[c, -s], [s, c]])
    worst = 0.0
    for rings in RINGS:
        for alpha in ALPHAS:
            for k in _k_points():
                for valley in (1, -1):
                    original = _spectrum(_h(alpha, rings, k, valley=valley))
                    rotated = _spectrum(_h(alpha, rings, rotation @ k, valley=valley))
                    worst = max(worst, np.abs(original - rotated).max())
    return CheckResult("Поворот на 120°", tolerance, worst, worst <= tolerance)


def check_operator_action() -> CheckResult:
    """Compare H c with the real-space operator acting on the same wave function."""
    tolerance = 1e-10
    phi = 0.7
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for rings in RINGS:
        for alpha in ALPHAS:
            for k in _k_points():
                for valley in (1, -1):
                    model = build_hamiltonian(
                        alpha * E0, rings=rings, k=k, phi=phi, L=L_NM, valley=valley
                    )
                    index = model.index
                    ring = np.maximum.reduce(
                        [np.abs(index["m"]), np.abs(index["n"]),
                         np.abs(index["m"] - index["n"])]
                    )
                    c = (
                        rng.standard_normal(len(index))
                        + 1j * rng.standard_normal(len(index))
                    ) * (ring <= rings - 1)
                    c /= np.linalg.norm(c)

                    q = k + _harmonic_vectors(index)
                    r = rng.uniform(-model.period_nm, model.period_nm, size=(N_R, 2))
                    waves = np.exp(1j * r @ q.T)

                    from_matrix = np.stack(
                        [waves @ (model.matrix @ c)[0::2],
                         waves @ (model.matrix @ c)[1::2]],
                        axis=1,
                    )

                    psi_a, psi_b = waves @ c[0::2], waves @ c[1::2]
                    px_a, px_b = (waves * q[:, 0]) @ c[0::2], (waves * q[:, 0]) @ c[1::2]
                    py_a, py_b = (waves * q[:, 1]) @ c[0::2], (waves * q[:, 1]) @ c[1::2]
                    potential = _potential(r, alpha * E0, phi)
                    from_operator = np.stack(
                        [
                            model.hbar_vf * (valley * px_b - 1j * py_b)
                            + potential * psi_a,
                            model.hbar_vf * (valley * px_a + 1j * py_a)
                            + potential * psi_b,
                        ],
                        axis=1,
                    )
                    worst = max(worst, np.abs(from_matrix - from_operator).max() / E0)
    return CheckResult("Действие оператора", tolerance, worst, worst <= tolerance)


def _oddness_deviation(alpha: float, rings: int, phi: float) -> float:
    worst = 0.0
    for k in _k_points():
        plus = _spectrum(_h(alpha, rings, -k, phi))
        minus = _spectrum(-_h(alpha, rings, k, phi))
        worst = max(worst, np.abs(plus - minus).max())
    return worst


def check_potential_oddness() -> CheckResult:
    tolerance = 1e-10
    control_threshold = 1e-3
    worst = max(
        _oddness_deviation(alpha, rings, PHI) for rings in RINGS for alpha in ALPHAS
    )
    control = min(_oddness_deviation(0.1, rings, 0.0) for rings in RINGS)
    return CheckResult(
        "Нечётность потенциала",
        tolerance,
        worst,
        worst <= tolerance and control > control_threshold,
        control_deviation=control,
        control_threshold=control_threshold,
    )


def velocity_coefficient(alpha: float, rings: int = 3, q_fraction: float = 1e-4) -> float:
    """Return (1 - v / v_F) / alpha^2 from the splitting of the two central bands."""
    q = q_fraction * G_NORM
    energies = _spectrum(_h(alpha, rings, (q, 0.0)))
    n_g = number_of_harmonics(rings)
    velocity_ratio = (energies[n_g] - energies[n_g - 1]) / (2 * q_fraction)
    return (1 - velocity_ratio) / alpha**2


def check_weak_potential() -> CheckResult:
    tolerance = 0.01
    worst = max(abs(velocity_coefficient(alpha) / 6 - 1) for alpha in (0.01, 0.02))
    return CheckResult("Слабый потенциал", tolerance, worst, worst <= tolerance)


def check_periodicity() -> CheckResult:
    tolerance = 1e-3
    alpha, rings = 0.3, 8
    n_g = number_of_harmonics(rings)
    shift = np.array([G_NORM, 0.0])
    worst = 0.0
    for k in _k_points():
        base = central_bands(_spectrum(_h(alpha, rings, k)), n_g)
        shifted = central_bands(_spectrum(_h(alpha, rings, k + shift)), n_g)
        worst = max(worst, np.abs(base - shifted).max())
    return CheckResult("Периодичность по мини-зоне", tolerance, worst, worst <= tolerance)


CHECKS = (
    check_hermiticity,
    check_free_limit,
    check_coupling_structure,
    check_rotation,
    check_operator_action,
    check_potential_oddness,
    check_weak_potential,
    check_periodicity,
)
