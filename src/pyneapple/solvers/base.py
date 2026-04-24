"""Base solver interface for optimization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Any

import numpy as np
from loguru import logger


@dataclass
class _PixelFitResult:
    """Internal per-pixel result produced by a solver's ``_fit_single_pixel``.

    This is a private implementation detail.  Consumer code should work with
    the public :class:`~pyneapple.result.FitResult` assembled by the fitter.

    Attributes:
        params: 1-D array of fitted parameter values for one pixel.
        covariance: Parameter covariance matrix ``(n_params, n_params)``, or
            ``None`` when not available (e.g. NNLS).
        success: ``True`` if the optimiser converged for this pixel.
        message: Optimiser status message, or ``None`` when not available.
        n_iterations: Number of optimiser iterations, or ``None`` when the
            backend does not expose this (e.g. ``curve_fit``).
        residual: Scalar residual norm for this pixel, or ``None`` when not
            available.
    """

    params: np.ndarray
    covariance: np.ndarray | None = None
    success: bool = True
    message: str | None = None
    n_iterations: int | None = None
    residual: float | None = None


class BaseSolver(ABC):
    """Abstract base class for optimization solvers."""

    def __init__(
        self,
        model: Any,
        max_iter: int = 250,
        tol: float = 1e-8,
        verbose: bool = False,
        **solver_kwargs,
    ):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.diagnostics_: dict[str, Any] = {}
        self.params_: dict[str, Any] = {}
        self.pixel_results_: list[_PixelFitResult] = []

        if self.verbose:
            logger.info(
                f"Initialized {self.__class__.__name__} with solver_kwargs={solver_kwargs}"
            )

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseSolver":
        """Fit the optimization model."""
        return self

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics information about the solver."""
        if len(self.diagnostics_) == 0:
            error_msg = "No diagnostics available. Ensure fit() has been called and diagnostics are stored."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return self.diagnostics_.copy()

    def get_params(self) -> dict[str, Any]:
        """Return the fitted parameters."""
        if len(self.params_) == 0:
            error_msg = "No parameters available. Ensure fit() has been called and parameters are stored."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return self.params_.copy()

    def _reset_state(self):
        """Reset solver state before a new fit."""
        self.diagnostics_ = {}
        self.params_ = {}
        self.pixel_results_ = []
