"""Non-Negative Least Squares (NNLS) solvers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import nnls
from tqdm import tqdm
from joblib import Parallel, delayed

from loguru import logger
from .base import BaseSolver
from ..models.base import DistributionModel
from ..model_functions.nnls import regularization_matrix


class NNLSSolver(BaseSolver):
    """Non-Negative Least Squares (NNLS) solver.

    The bin grid and basis matrix are owned by the ``model`` (a
    :class:`~pyneapple.models.base.DistributionModel`), keeping physics separate
    from the optimization backend.

    Args:
        model: Distribution model that provides ``bins`` and ``get_basis()``.
        reg_order: Regularization order (0=none, 1=first diff, 2=second diff,
                   3=extended second diff) (default: 0)
        mu: Regularization strength parameter (default: 0.02)
        max_iter: Maximum NNLS iterations (default: 250)
        verbose: Enable verbose logging (default: False)
    """

    def __init__(
        self,
        model: DistributionModel,
        reg_order: int = 0,
        mu: float = 0.02,
        max_iter: int = 250,
        tol: float = 1e-8,
        verbose=False,
        multi_threading: bool = False,
        **solver_kwargs: Any,
    ) -> None:
        super().__init__(model, max_iter, tol, verbose, **solver_kwargs)
        self.multi_threading = multi_threading
        self.n_pools = solver_kwargs.pop(
            "n_pools", None
        )  # Number of parallel pools for multithreading
        self.reg_order = reg_order
        self.mu = mu

    def get_regularization_matrix(self) -> np.ndarray:
        """Construct the regularization matrix based on the specified order.

        Returns:
            Regularization matrix of shape (n_bins, n_bins)
        """
        return regularization_matrix(self.model.n_bins, self.reg_order, self.mu)

    def _build_regularized_basis(self, xdata: np.ndarray) -> np.ndarray:
        """Construct the regularized basis matrix for NNLS.

        Args:
            xdata: B-values, shape (n_measurements,)
        Returns:
            Regularized basis matrix of shape (n_measurements + n_bins, n_bins)
        """
        basis = self.model.get_basis(xdata)  # shape (n_measurements, n_bins)
        reg_matrix = self.get_regularization_matrix()  # shape (n_bins, n_bins)
        return np.concatenate(
            [basis, reg_matrix], axis=0
        )  # shape (n_measurements + n_bins, n_bins)

    def _extend_signal(self, signal: np.ndarray) -> np.ndarray:
        """Extend the signal vector to match the regularized basis.

        Args:
            signal: Original signal vector of shape (n_pixels, n_measurements)

        Returns:
            Extended signal vector of shape (n_pixels, n_measurements + n_bins)
        """
        return np.concatenate(
            (signal, np.zeros((signal.shape[0], self.model.n_bins))), axis=1
        )

    def fit(
        self,
        xdata: np.ndarray,
        signal: np.ndarray,
        pixel_fixed_params: dict[str, np.ndarray] | None = None,
    ) -> "NNLSSolver":
        """Fit the NNLS model to the data.

        Args:
            xdata: B-values, shape (n_measurements,)
            signal: Signal vector, shape (n_pixels, n_measurements)
            pixel_fixed_params: Ignored for NNLS solver (accepted for API
                compatibility with :class:`CurveFitSolver`).

        Returns:
            Self with fitted parameters and diagnostics.
        """
        self._reset_state()

        if self.verbose:
            logger.info(
                f"Fitting NNLSSolver with reg_order={self.reg_order}, mu={self.mu}, max_iter={self.max_iter}"
            )

        # Build the regularized basis and extended signal
        regularized_basis = self._build_regularized_basis(xdata)
        if signal.ndim == 1:
            signal = signal[
                np.newaxis, :
            ]  # reshape to (1, n_measurements) if signal is 1D
        extended_signal = self._extend_signal(signal)
        self.n_pixels = signal.shape[0]

        # Solve the NNLS problem
        coeffs, residual = self._fit_data(regularized_basis, extended_signal)
        self.params_["coefficients"] = coeffs  # shape (n_pixels, n_bins)
        self.diagnostics_["residual"] = residual  # shape (n_pixels,)

        logger.info(f"NNLS fitting completed for {self.n_pixels} voxel(s)")
        return self

    def _fit_data(
        self, basis: np.ndarray, signal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Internal method to fit the NNLS model to the data.

        Args:
            basis: Regularized basis matrix, shape (n_measurements + n_bins, n_bins)
            signal: Signal vector, shape (n_pixels, n_measurements + n_bins)

        Returns:
            tuple[np.ndarray, np.ndarray]: Coefficients of shape
                ``(n_pixels, n_bins)`` and residuals of shape ``(n_pixels,)``.
        """

        if self.multi_threading and self.n_pixels > 1:
            # Determine number of threads to use
            n_jobs = self.n_pools if self.n_pools > 0 else -1
            logger.info(
                f"Using {n_jobs if n_jobs > 0 else 'all'} CPU cores for parallel fitting"
            )

            # Run parallel fits with progress bar
            results = Parallel(n_jobs=n_jobs, verbose=10 if self.verbose else 0)(
                delayed(self._fit_single_pixel)(
                    basis,
                    signal[i],
                    pixel_idx=i,
                )
                for i in range(self.n_pixels)
            )

            # Unpack results
            coeffs_list = [
                r[0] if r is not None else np.full(self.model.n_bins, np.nan)
                for r in results
            ]
            residuals_list = [
                r[1] if r is not None else np.nan for r in results
            ]  # Extract residuals
        else:
            iterator = tqdm(
                range(self.n_pixels), desc="Fitting pixels", disable=not self.verbose
            )
            coeffs_list, residuals_list = [], []
            for i in iterator:
                coeff, residual = self._fit_single_pixel(basis, signal[i], pixel_idx=i)
                coeffs_list.append(coeff)
                residuals_list.append(residual)  # Store residual for diagnostics

        return np.array(coeffs_list), np.array(
            residuals_list
        )  # shape (n_pixels, n_bins), shape (n_pixels,)

    def _fit_single_pixel(
        self, basis: np.ndarray, signal: np.ndarray, pixel_idx: int
    ) -> tuple[np.ndarray, float]:
        """Fit NNLS for a single pixel.

        Args:
            basis: Regularized basis matrix, shape (n_measurements + n_bins, n_bins)
            signal: Signal vector for the pixel, shape (n_measurements + n_bins,)
            pixel_idx: Index of the pixel being fitted (for logging)
        Returns:
            Coefficients of shape (n_bins,) for the pixel.
        """
        try:
            coeffs, residual = nnls(
                basis, signal, maxiter=self.max_iter
            )  # shape (n_bins,), scalar residual
            return coeffs, residual
        except Exception as e:
            logger.warning(
                f"Pixel {pixel_idx} fit failed: Error occurred while fitting NNLS: {e}"
            )
            return np.zeros(self.model.n_bins), float(np.linalg.norm(signal))
