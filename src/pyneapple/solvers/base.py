"""Base solver interface for optimization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from ..result import FitResult


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
    def fit(self, *args, **kwargs) -> BaseSolver:
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

    # ------------------------------------------------------------------
    # Solver-level FitResult
    # ------------------------------------------------------------------

    @property
    def result_(self) -> FitResult | None:
        """Partial :class:`~pyneapple.result.FitResult` assembled from per-pixel results.

        Available immediately after :meth:`fit` returns.  Contains all
        solver-level diagnostic fields (``params``, ``success``,
        ``covariance``, ``residuals``, ``n_iterations``, ``messages``,
        ``n_pixels``, ``solver_name``, ``model_name``) but **not** the
        fitter-specific fields (``r_squared``, ``fit_time``,
        ``image_shape``, ``pixel_indices``), which are added by the
        fitter's ``_assemble_fit_result()``.

        Returns:
            A :class:`~pyneapple.result.FitResult` if :meth:`fit` has
            been called, otherwise ``None``.
        """
        if not self.pixel_results_:
            return None
        return self._build_result()

    def _build_result(self) -> FitResult:
        """Assemble a partial :class:`~pyneapple.result.FitResult` from ``pixel_results_``.

        Called by :attr:`result_` every time the property is accessed
        (no internal caching — the result is always fresh).

        Returns:
            A :class:`~pyneapple.result.FitResult` with solver-level
            fields populated.
        """
        prs = self.pixel_results_
        n_pixels = len(prs)

        # --- success ---
        success = np.array([pr.success for pr in prs], dtype=bool)

        # --- n_iterations (None when all are None) ---
        iters = [pr.n_iterations for pr in prs]
        if any(it is not None for it in iters):
            n_iterations: np.ndarray | None = np.array(
                [it if it is not None else -1 for it in iters], dtype=np.intp
            )
        else:
            n_iterations = None

        # --- messages (None when all are None) ---
        msgs = [pr.message for pr in prs]
        messages: list[str | None] | None = (
            msgs if any(m is not None for m in msgs) else None
        )

        # --- covariance (None for NNLS which has no covariance) ---
        covs = [pr.covariance for pr in prs]
        if any(c is not None for c in covs):
            n_params = prs[0].params.shape[0]
            covariance: np.ndarray | None = np.array(
                [
                    c if c is not None else np.full((n_params, n_params), np.nan)
                    for c in covs
                ]
            )
        else:
            covariance = None

        # --- residuals (None when all are None) ---
        residuals_list = [pr.residual for pr in prs]
        if any(r is not None for r in residuals_list):
            residuals: np.ndarray | None = np.array(
                [r if r is not None else np.nan for r in residuals_list],
                dtype=np.float64,
            )
        else:
            residuals = None

        return FitResult(
            params=dict(self.params_),
            success=success,
            n_iterations=n_iterations,
            messages=messages,
            covariance=covariance,
            residuals=residuals,
            n_pixels=n_pixels,
            solver_name=self.__class__.__name__,
            model_name=self.model.__class__.__name__,
        )
