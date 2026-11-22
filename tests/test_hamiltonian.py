import numpy as np
import pytest

from moire_dirac import build_hamiltonian, mini_bz_path
from moire_dirac.checks import CHECKS


@pytest.mark.parametrize("rings, size", [(3, 74), (8, 434)])
def test_interface(rings, size):
    by_period = build_hamiltonian(0.02, rings=rings, k=(0.03, -0.01), L=10.0, as_tensor=True)
    by_angle = build_hamiltonian(
        0.02, rings=rings, k=(0.03, -0.01), theta=0.025, a=0.25, as_tensor=True
    )
    n_g = size // 2

    assert by_period.matrix.shape == (size, size)
    assert len(by_period.index) == size
    assert by_period.tensor.shape == (n_g, 2, n_g, 2)
    assert np.abs(by_period.matrix - by_period.matrix.conj().T).max() <= 1e-12 * by_period.e0
    assert np.abs(by_period.matrix - by_angle.matrix).max() <= 1e-12 * by_period.e0

    row, column = 5, size - 2
    element = by_period.tensor[row // 2, row % 2, column // 2, column % 2]
    assert element == by_period.matrix[row, column]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rings": 3},
        {"rings": 3, "L": 10.0, "theta": 0.025, "a": 0.25},
        {"rings": 3, "theta": 0.025},
        {"rings": 0, "L": 10.0},
        {"rings": 3, "L": 10.0, "valley": 0},
        {"rings": 3, "L": 10.0, "k": (0.0, 0.0, 0.0)},
    ],
)
def test_invalid_input(kwargs):
    with pytest.raises(ValueError):
        build_hamiltonian(0.02, **kwargs)


def test_path_has_200_points_and_exact_corners():
    path = mini_bz_path(1.0)
    assert len(path) == 200
    assert np.array_equal(path[0], [0.0, 0.0])
    assert np.array_equal(path[-1], [0.0, 0.0])
    assert np.isclose(np.linalg.norm(path - [0.5, 0.5 / np.sqrt(3)], axis=1).min(), 0.0)
    assert np.isclose(np.linalg.norm(path - [0.5, 0.0], axis=1).min(), 0.0)


@pytest.mark.parametrize("check", CHECKS, ids=lambda check: check.__name__)
def test_correctness_check(check):
    result = check()
    assert result.passed, result
