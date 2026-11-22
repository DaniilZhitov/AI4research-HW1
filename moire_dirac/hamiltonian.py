"""Plane-wave Hamiltonian of one graphene valley in a first-harmonic moire potential.

The model is

    H_xi = hbar v_F (xi sigma_x k_x + sigma_y k_y) + V(r) sigma_0,
    V(r) = 2 V_1 sum_{j=1,3,5} cos(G_j . r + phi),

with G_j = |G| (cos((j-1) pi/3), sin((j-1) pi/3)) and |G| = 4 pi / (sqrt(3) L).
Basis states are |k + G, s> with G = m b_1 + n b_2, b_1 = G_1, b_2 = G_3 and
s in {A, B}. Energies are in eV, lengths in nm and wave vectors in 1/nm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

HBAR_EV_S = 6.582119569e-16
NM_PER_M = 1e9

# (m, n) coordinates of G_1 ... G_6.
FIRST_SHELL = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))

INDEX_DTYPE = np.dtype([("row", "i4"), ("m", "i4"), ("n", "i4"), ("sublattice", "U1")])


@dataclass(frozen=True)
class MoireHamiltonian:
    """Matrix of one valley together with its basis labels and energy scale."""

    matrix: np.ndarray  # complex Hermitian, shape (2 N_G, 2 N_G), eV
    index: np.ndarray  # structured array: row -> (m, n, sublattice)
    n_harmonics: int
    period_nm: float
    g_norm: float  # |G|, 1/nm
    hbar_vf: float  # eV nm
    e0: float  # hbar v_F |G|, eV
    tensor: np.ndarray | None = None  # H[G, s, G', s'], shape (N_G, 2, N_G, 2)


def number_of_harmonics(rings: int) -> int:
    """Return N_G = 1 + 3 N (N + 1) for a hexagonal cutoff of N rings."""
    return 1 + 3 * rings * (rings + 1)


@lru_cache(maxsize=None)
def _basis(rings: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return harmonics and the (row, column, shell) lists of coupled harmonics."""
    nodes = [
        (max(abs(m), abs(n), abs(m - n)), m, n)
        for m in range(-rings, rings + 1)
        for n in range(-rings, rings + 1)
        if max(abs(m), abs(n), abs(m - n)) <= rings
    ]
    nodes.sort()
    harmonics = np.array([(m, n) for _, m, n in nodes], dtype=int)
    position = {(m, n): i for i, (_, m, n) in enumerate(nodes)}

    rows, cols, shells = [], [], []
    for row, (m, n) in enumerate(harmonics.tolist()):
        for shell, (dm, dn) in enumerate(FIRST_SHELL):
            col = position.get((m - dm, n - dn))
            if col is not None:
                rows.append(row)
                cols.append(col)
                shells.append(shell)

    arrays = (
        harmonics,
        np.array(rows, dtype=int),
        np.array(cols, dtype=int),
        np.array(shells, dtype=int),
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _resolve_period(L: float | None, theta: float | None, a: float | None) -> float:
    if L is not None:
        if theta is not None or a is not None:
            raise ValueError("Pass either L or (theta, a), not both.")
        period = float(L)
    else:
        if theta is None or a is None:
            raise ValueError("Pass L or both theta and a.")
        if theta <= 0 or a <= 0:
            raise ValueError("theta and a must be positive.")
        period = float(a) / float(theta)
    if period <= 0:
        raise ValueError("The moire period must be positive.")
    return period


def build_hamiltonian(
    V1: float,
    *,
    rings: int,
    k: tuple[float, float] | np.ndarray = (0.0, 0.0),
    phi: float = np.pi / 2,
    L: float | None = None,
    theta: float | None = None,
    a: float | None = None,
    v_F: float = 1e6,
    valley: int = 1,
    as_tensor: bool = False,
) -> MoireHamiltonian:
    """Build the plane-wave matrix of the valley ``valley`` at the point ``k``.

    Args:
        V1: first-harmonic amplitude, eV.
        rings: number of hexagonal rings N, at least 1.
        k: wave vector measured from the Dirac point, 1/nm.
        phi: first-harmonic phase, radians.
        L: moire period, nm. Alternative to ``theta`` and ``a``.
        theta: twist angle, radians; the period is then L = a / theta.
        a: hBN lattice constant, nm.
        v_F: Fermi velocity, m/s.
        valley: +1 for K, -1 for K'.
        as_tensor: also return H[G, s, G', s'].
    """
    if valley not in (1, -1):
        raise ValueError("valley must be +1 or -1.")
    if int(rings) != rings or rings < 1:
        raise ValueError("rings must be a positive integer.")
    k = np.asarray(k, dtype=float)
    if k.shape != (2,):
        raise ValueError("k must be a real two-component vector.")

    rings = int(rings)
    period = _resolve_period(L, theta, a)
    g_norm = 4 * np.pi / (np.sqrt(3) * period)
    hbar_vf = HBAR_EV_S * v_F * NM_PER_M
    e0 = hbar_vf * g_norm

    harmonics, rows, cols, shells = _basis(rings)
    n_g = len(harmonics)
    b1 = g_norm * np.array([1.0, 0.0])
    b2 = g_norm * np.array([-0.5, np.sqrt(3) / 2])
    q = k + harmonics[:, 0, None] * b1 + harmonics[:, 1, None] * b2

    matrix = np.zeros((2 * n_g, 2 * n_g), dtype=complex)

    # Dirac block, diagonal in the harmonics.
    diagonal = np.arange(n_g)
    ab = hbar_vf * (valley * q[:, 0] - 1j * q[:, 1])
    matrix[2 * diagonal, 2 * diagonal + 1] = ab
    matrix[2 * diagonal + 1, 2 * diagonal] = ab.conj()

    # Scalar coupling V_{G - G'}: e^{+i phi} on G_1, G_3, G_5, e^{-i phi} on the rest.
    phases = np.where(shells % 2 == 0, np.exp(1j * phi), np.exp(-1j * phi))
    coupling = V1 * phases
    matrix[2 * rows, 2 * cols] = coupling
    matrix[2 * rows + 1, 2 * cols + 1] = coupling

    index = np.empty(2 * n_g, dtype=INDEX_DTYPE)
    index["row"] = np.arange(2 * n_g)
    index["m"] = np.repeat(harmonics[:, 0], 2)
    index["n"] = np.repeat(harmonics[:, 1], 2)
    index["sublattice"] = np.tile(["A", "B"], n_g)

    tensor = matrix.reshape(n_g, 2, n_g, 2).copy() if as_tensor else None
    return MoireHamiltonian(
        matrix=matrix,
        index=index,
        n_harmonics=n_g,
        period_nm=period,
        g_norm=g_norm,
        hbar_vf=hbar_vf,
        e0=e0,
        tensor=tensor,
    )


def central_bands(energies: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Return the bands N_G - 3 ... N_G + 4 (counted from one) of a sorted spectrum."""
    return energies[..., n_harmonics - 4 : n_harmonics + 4]


def mini_bz_path(g_norm: float, n_points: int = 200) -> np.ndarray:
    """Return the closed path Gamma - K_m - M_m - Gamma with exact corner points."""
    corners = g_norm * np.array(
        [[0.0, 0.0], [0.5, 0.5 / np.sqrt(3)], [0.5, 0.0], [0.0, 0.0]]
    )
    lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    intervals = n_points - 1
    counts = np.floor(intervals * lengths / lengths.sum()).astype(int)
    remainders = intervals * lengths / lengths.sum() - counts
    for segment in np.argsort(-remainders)[: intervals - counts.sum()]:
        counts[segment] += 1

    pieces = [
        np.linspace(corners[i], corners[i + 1], counts[i], endpoint=False)
        for i in range(3)
    ]
    return np.vstack([*pieces, corners[-1:]])


def sample_mini_bz(g_norm: float, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Draw points uniformly from the hexagonal mini Brillouin zone."""
    angles = np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    radius = g_norm / np.sqrt(3)
    points: list[np.ndarray] = []
    while len(points) < n_points:
        candidate = rng.uniform(-radius, radius, size=2)
        if np.all(normals @ candidate <= g_norm / 2):
            points.append(candidate)
    return np.array(points)
