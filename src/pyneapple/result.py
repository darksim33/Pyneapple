"""Public result container returned by all fitters after fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FitResult:
    """Spatial result of a fitter.fit() call.

    Returned by every concrete :class:`~pyneapple.fitters.base.BaseFitter`
    subclass and stored as ``fitter.results_`` after calling ``fit()``.

    Attributes:
        params: Fitted parameter maps.  Keys are model parameter names; each
            value is a 1-D array of shape ``(n_pixels,)`` with the per-pixel
            fitted value.
        success: Boolean array of shape ``(n_pixels,)`` indicating whether the
            optimiser converged for each pixel.
        n_iterations: Integer array of shape ``(n_pixels,)`` with the number
            of optimiser iterations per pixel.  ``None`` when the backend does
            not expose iteration counts (e.g. ``CurveFitSolver``).
        messages: Per-pixel optimiser status messages.  ``None`` when no
            backend provides them.
        covariance: Covariance matrix array of shape
            ``(n_pixels, n_params, n_params)``.  ``None`` for NNLS fits.
        residuals: Raw per-pixel residual norms of shape ``(n_pixels,)``.
            ``None`` when unavailable.
        r_squared: Per-pixel R² quality metric of shape ``(n_pixels,)``.
            Computed from predicted vs. observed signal after fitting.
            ``None`` when computation fails.
        fit_time: Total wall-clock time in seconds for the ``fit()`` call.
        image_shape: Shape of the original 4-D image ``(X, Y, Z, N)`` passed
            to ``fit()``.  ``None`` when not available.
        pixel_indices: Spatial ``(x, y, z)`` coordinate for each fitted pixel.
            ``None`` when not available.
        n_pixels: Number of pixels that were fitted.
        solver_name: Class name of the solver used (e.g. ``"CurveFitSolver"``).
        model_name: Class name of the model used (e.g. ``"BiExpModel"``).
    """

    # --- Core ---
    params: dict[str, np.ndarray]

    # --- Convergence / success ---
    success: np.ndarray
    n_iterations: np.ndarray | None = None
    messages: list[str | None] | None = None

    # --- Diagnostics ---
    covariance: np.ndarray | None = None
    residuals: np.ndarray | None = None
    r_squared: np.ndarray | None = None

    # --- Timing ---
    fit_time: float = 0.0

    # --- Metadata ---
    image_shape: tuple | None = None
    pixel_indices: list[tuple] | None = None
    n_pixels: int = 0
    solver_name: str = ""
    model_name: str = ""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_converged(self) -> int:
        """Number of pixels where the optimiser converged."""
        return int(np.sum(self.success))

    @property
    def convergence_rate(self) -> float:
        """Fraction of pixels where the optimiser converged (0.0 – 1.0)."""
        if self.n_pixels == 0:
            return 0.0
        return float(self.n_converged / self.n_pixels)

    @property
    def mean_r_squared(self) -> float | None:
        """Mean R² across all pixels.  ``None`` if R² was not computed."""
        if self.r_squared is None:
            return None
        if np.all(np.isnan(self.r_squared)):
            return float("nan")
        return float(np.nanmean(self.r_squared))

    def __repr__(self) -> str:  # pragma: no cover
        r2_str = (
            f"{self.mean_r_squared:.4f}" if self.mean_r_squared is not None else "n/a"
        )
        return (
            f"FitResult("
            f"model={self.model_name!r}, "
            f"solver={self.solver_name!r}, "
            f"n_pixels={self.n_pixels}, "
            f"converged={self.n_converged}/{self.n_pixels}, "
            f"mean_R²={r2_str}, "
            f"fit_time={self.fit_time:.3f}s)"
        )
