"""Base fitter interface for spatial model fitting orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np

from loguru import logger

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

    Examples
    --------

    """

    def __init__(self, solver: BaseSolver, **fitter_kwargs):
        """Initialize the fitter with a model, solver, and any additional configuration.

        Args:
            solver: An instance of a BaseSolver subclass responsible
                for the optimization routine.
            **fitter_kwargs: Additional keyword arguments for fitter configuration.
        """
        self.solver = solver
        self.fitter_kwargs = fitter_kwargs

        # Result storage set by fit() method
        self.results_: Any = None  # To store fitted parameters and metadata
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
            image: 4D array of shape (X, Y, Z, N) where N is the number of measurements (e.g., b-values).
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
        pass

    def get_fitted_params(self) -> dict[str, np.ndarray] | None:
        """Get the fitted parameters in a structured format.

        Returns:
            dict[str, np.ndarray] | None: Dictionary of parameter names to arrays of fitted values, or None if not fitted.
        """
        return self.fitted_params_

    def _extract_pixel_data(
        self, image: np.ndarray, segmentation: np.ndarray
    ) -> np.ndarray:
        """Extract pixel data from the input array and store pixel indices.

        Args:
            image: Array of signal intensities.
            segmentation: Array of binary mask for pixels to fit.
        Returns:
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
