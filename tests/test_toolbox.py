"""Shared test helpers for the Pyneapple test suite.

Plain importable functions and constants — not pytest fixtures — so they can be
reused across any test module without coupling to a specific conftest scope.
"""

from __future__ import annotations

import numpy as np

from pyneapple.models import MonoExpModel
from pyneapple.solvers import CurveFitSolver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B_VALUES = np.array(
    [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200],
    dtype=float,
)
"""Standard 16-point b-value array used across tests."""

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def make_monoexp_image(
    n_x: int = 4,
    n_y: int = 4,
    n_z: int = 1,
    S0: float = 1000.0,
    D: float = 0.001,
    b_values: np.ndarray = B_VALUES,
) -> np.ndarray:
    """Create a noise-free 4-D monoexponential DWI image.

    Returns an array of shape ``(n_x, n_y, n_z, len(b_values))`` where every
    voxel contains the same monoexponential signal.
    """
    signal = MonoExpModel().forward(b_values, S0, D)  # (n_b,)
    return np.tile(signal, (n_x, n_y, n_z, 1))


# ---------------------------------------------------------------------------
# Solver / fitter helpers
# ---------------------------------------------------------------------------


def make_monoexp_solver(
    p0: dict | None = None,
    bounds: dict | None = None,
) -> CurveFitSolver:
    """Return a CurveFitSolver backed by MonoExpModel with sensible defaults.

    Args:
        p0: Initial parameter guesses. Defaults to ``{"S0": 1000.0, "D": 0.001}``.
        bounds: Parameter bounds. Defaults to ``{"S0": (1.0, 5000.0), "D": (1e-5, 0.1)}``.
    """
    if p0 is None:
        p0 = {"S0": 1000.0, "D": 0.001}
    if bounds is None:
        bounds = {"S0": (1.0, 5000.0), "D": (1e-5, 0.1)}
    return CurveFitSolver(
        model=MonoExpModel(),
        max_iter=250,
        tol=1e-8,
        p0=p0,
        bounds=bounds,
    )


# ---------------------------------------------------------------------------
# IDEAL-specific helpers
# ---------------------------------------------------------------------------


def make_dim_steps(full_spatial_shape: tuple[int, int]) -> np.ndarray:
    """Return a ``(2, 2)`` dim_steps array for a 2-step IDEAL run.

    The first step uses half the spatial resolution; the second step uses the
    full resolution given by *full_spatial_shape*.

    Args:
        full_spatial_shape: ``(n_x, n_y)`` at full resolution.

    Returns:
        np.ndarray of shape ``(2, 2)`` with monotonically increasing values
        along each row, ending at *full_spatial_shape*.

    Examples
    --------
    >>> make_dim_steps((4, 4))
    array([[2, 4],
           [2, 4]])
    """
    half_x = max(2, full_spatial_shape[0] // 2)
    half_y = max(2, full_spatial_shape[1] // 2)
    return np.array([[half_x, full_spatial_shape[0]], [half_y, full_spatial_shape[1]]])
