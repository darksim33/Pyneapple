"""Base fitter interface for spatial model fitting orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from loguru import logger

from ..result import FitResult

if TYPE_CHECKING:
    from ..solvers.base import BaseSolver


class BaseFitter(ABC):
    """Abstract base class for all fitters.

    Fitters orchestrate spatial fitting workflows, using solvers and managing
    the iteration over pixels, ROIs, or other spatial units. They provide the
    high-level scikit-learn style fit()/predict() API.

    Architecture:
        1. Fitter accepts model configuration and fitting strategy
        2. Fitter accepts appropriate Solver instance(s)
        3. fit() method orchestrates data preparation and solver calls
        4. predict() uses stored fitted parameters to generate predictions
        5. results_ attribute provides rich export capabilities

    Fitter Types:
        - PixelWiseFitter: Fits each pixel independently

    Examples:

    """

    def __init__(self, solver: BaseSolver, verbose: bool = False, **fitter_kwargs):
        """Initialize the fitter with a model, solver, and any additional configuration.

        Args:
            solver: An instance of a BaseSolver subclass responsible
                for the optimization routine.
            **fitter_kwargs: Additional keyword arguments for fitter configuration.
        """
        self.solver = solver
        self.verbose = verbose
        self.fitter_kwargs = fitter_kwargs

        # Result storage set by fit() method
        self.results_: FitResult | None = None
        self.fitted_params_: dict = {}  # To store fitted parameters in a structured format

        # Image Metadata
        self.image_shape: tuple | None = None
        self.pixel_indices: list[tuple] | None = (
            None  # To store pixel indices for mapping results back to image space
        )
        self.n_measurements: int | None = (
            None  # To store number of measurements (e.g., b-values)
        )

        logger.debug(
            f"Initialized {self.__class__.__name__} with model={solver.model.__class__.__name__}"
        )

    @abstractmethod
    def fit(
        self,
        xdata: np.ndarray,
        image: np.ndarray,
        segmentation: np.ndarray | None = None,
        **fit_kwargs,
    ) -> "BaseFitter":
        """Fit the model to the data.

        Args:
            xdata: 1D array of independent variable (e.g., b-values).
            image: 4D array of shape (X, Y, Z, N) where N is the number of
                measurements (e.g., b-values).
            segmentation: Optional array of shape (X, Y, Z) used to restrict
                fitting to non-zero voxels. If ``None``, all voxels are fitted.
            **fit_kwargs: Additional keyword arguments for fitting.

        Returns:
            self: Returns the fitted instance for chaining.
        """
        return self

    @abstractmethod
    def predict(self, xdata: np.ndarray, **predict_kwargs) -> np.ndarray:
        """Generate predictions from the fitted model.

        Args:
            xdata: 1D array of independent variable (e.g., b-values).
            **predict_kwargs: Additional keyword arguments for prediction.

        Returns:
            np.ndarray: Array of predicted values.
        """
        self._check_fitted()  # Ensure fit() has been called
        if xdata.ndim != 1:
            raise ValueError(
                f"xdata must be a 1D array of independent variable values, got shape {xdata.shape}."
            )

        # Collect params in model param oder: shape (n_parms, n_pixels)
        param_names = self._get_param_names()

        results = np.stack(
            [self.fitted_params_[param] for param in param_names], axis=0
        )  # shape (n_params, n_pixels)
        n_pixels = results.shape[1]

        predictions = np.empty((n_pixels, len(xdata)), dtype=np.float64)
        for idx in tqdm(
            range(n_pixels),
            total=n_pixels,
            desc="Predicting : ",
            disable=not self.verbose,
        ):
            predictions[idx] = self.solver.model.forward(xdata, *results[:, idx])

        # Reconstruct image shape: (X, Y, Z, N)
        output_shape = self.image_shape[:-1] + (xdata.size,)  # type: ignore handled by _check_fitted
        return self._reconstruct_volume(predictions, self.pixel_indices, output_shape)  # type: ignore handled by _check_fitted

    def get_fitted_params(self) -> dict[str, np.ndarray] | None:
        """Get the fitted parameters in a structured format.

        Returns:
            dict[str, np.ndarray] | None: Dictionary of parameter names to arrays of fitted values, or None if not fitted.
        """
        return self.fitted_params_

    # ------------------------------------------------------------------
    # FitResult assembly helpers
    # ------------------------------------------------------------------

    def _compute_r_squared(
        self,
        xdata: np.ndarray,
        pixel_signals: np.ndarray,
    ) -> np.ndarray:
        """Compute per-pixel R² from fitted parameters and observed signals.

        For parametric solvers (CurveFitSolver / ConstrainedCurveFitSolver),
        predictions are computed via ``model.forward(xdata, *params_for_pixel)``.
        For distribution solvers (NNLSSolver), predictions are computed as
        ``model.get_basis(xdata) @ coefficients.T``.

        Args:
            xdata: 1-D b-value array, shape ``(n_measurements,)``.
            pixel_signals: Observed signal matrix, shape ``(n_pixels, n_measurements)``.

        Returns:
            R² array of shape ``(n_pixels,)``.  Values are ``NaN`` where
            ``SS_tot == 0`` (constant signal).
        """
        from ..solvers.nnls_solver import NNLSSolver

        n_pixels = pixel_signals.shape[0]
        try:
            if isinstance(self.solver, NNLSSolver):
                basis = self.solver.model.get_basis(xdata)  # (n_measurements, n_bins)
                coeffs = self.solver.params_["coefficients"]  # (n_pixels, n_bins)
                predictions = coeffs @ basis.T  # (n_pixels, n_measurements)
            else:
                param_names = self.solver.model.param_names
                params_arr = np.stack(
                    [self.solver.params_[n] for n in param_names], axis=0
                )  # (n_params, n_pixels)
                predictions = np.empty((n_pixels, len(xdata)), dtype=np.float64)
                for i in range(n_pixels):
                    predictions[i] = self.solver.model.forward(xdata, *params_arr[:, i])

            ss_res = np.sum((pixel_signals - predictions) ** 2, axis=1)
            mean_signal = pixel_signals.mean(axis=1, keepdims=True)
            ss_tot = np.sum((pixel_signals - mean_signal) ** 2, axis=1)
            r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
            return r2.astype(np.float64)
        except Exception as exc:
            logger.warning(f"R² computation failed: {exc}")
            return np.full(n_pixels, np.nan, dtype=np.float64)

    def _assemble_fit_result(
        self,
        xdata: np.ndarray,
        pixel_signals: np.ndarray,
        fit_time: float,
        pixel_indices: list[tuple] | None = None,
    ) -> FitResult:
        """Build a :class:`~pyneapple.result.FitResult` from the solver state.

        Should be called immediately after ``solver.fit()`` completes,
        before any state is modified.

        Args:
            xdata: 1-D independent variable (b-values), shape ``(n_measurements,)``.
            pixel_signals: Observed signals used during fitting,
                shape ``(n_pixels, n_measurements)``.
            fit_time: Total wall-clock time in seconds for the fit.
            pixel_indices: Spatial coordinates for each pixel.  Defaults to
                ``self.pixel_indices`` when ``None``.

        Returns:
            Populated :class:`~pyneapple.result.FitResult`.
        """
        prs = self.solver.pixel_results_
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

        # --- residuals ---
        residuals_list = [pr.residual for pr in prs]
        if any(r is not None for r in residuals_list):
            residuals: np.ndarray | None = np.array(
                [r if r is not None else np.nan for r in residuals_list],
                dtype=np.float64,
            )
        else:
            residuals = None

        # --- R² ---
        r_squared = self._compute_r_squared(xdata, pixel_signals)

        return FitResult(
            params=dict(self.solver.params_),
            success=success,
            n_iterations=n_iterations,
            messages=messages,
            covariance=covariance,
            residuals=residuals,
            r_squared=r_squared,
            fit_time=fit_time,
            image_shape=self.image_shape,
            pixel_indices=pixel_indices
            if pixel_indices is not None
            else self.pixel_indices,
            n_pixels=n_pixels,
            solver_name=self.solver.__class__.__name__,
            model_name=self.solver.model.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_pixel_data(
        self, image: np.ndarray, segmentation: np.ndarray
    ) -> np.ndarray:
        """Extract pixel data from the input array and store pixel indices.

        Args:
            image: Array of signal intensities.
            segmentation: Binary mask array; non-zero voxels are fitted.

        Returns:
            np.ndarray: Pixel data of shape (n_pixels, n_measurements) for
                the masked voxels, with ``self.pixel_indices`` set.
        """
        if self.n_measurements is None:
            raise RuntimeError(
                "n_measurements must be set before extracting pixel data. Validate image data first."
            )
        if segmentation is not None:
            pixel_to_fit = image[segmentation != 0]
            pixel_positions = list(zip(*np.where(segmentation != 0)))
        else:
            pixel_to_fit = image.reshape(-1, self.n_measurements)
            pixel_positions = [
                tuple(np.unravel_index(idx, image.shape[:-1]))
                for idx in range(image.size // self.n_measurements)
            ]

        self.pixel_indices = pixel_positions  # Store pixel indices for mapping results back to image space
        return pixel_to_fit  # shape (n_pixels, n_measurements)

    def _reconstruct_volume(
        self,
        flat_values: np.ndarray,
        pixel_indices: list[tuple],
        spatial_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct a spatial volume from flat per-pixel values and indices.

        Args:
            flat_values: 1-D array of shape ``(n_pixels,)``.
            pixel_indices: Spatial index tuple for each pixel.
            spatial_shape: Shape of the output volume (e.g. ``(X, Y, Z)``).

        Returns:
            Volume of the given *spatial_shape*, zero-filled where no pixel
            was fitted.
        """
        vol = np.zeros(spatial_shape, dtype=np.float64)
        idx = tuple(zip(*pixel_indices))
        vol[idx] = flat_values
        return vol

    def _check_fitted(self) -> None:
        """Check if fitter has been fitted.

        Raises:
            RuntimeError: If fit() has not been called
        """
        if not self.fitted_params_:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                f"Call fit() before predict() or get_fitted_params()."
            )
        if self.pixel_indices is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not extracted pixel data yet. "
                f"Call _extract_pixel_data() before predict() or get_fitted_params()."
            )

    def _get_param_names(self) -> list[str]:
        """Get the parameter names from the solver's model."""
        return self.solver.model.param_names
