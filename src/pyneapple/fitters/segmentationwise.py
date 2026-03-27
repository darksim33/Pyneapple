"""Segmentation Wise Fitter for fitting of mean segmentation values."""

from loguru import logger

import numpy as np

from .base import BaseFitter
from ..solvers import CurveFitSolver
from ..utility.validation import (
    validate_xdata,
    validate_data_shapes,
    validate_segmentation,
    validate_fixed_param_maps,
)


class SegmentationWiseFitter(BaseFitter):
    """Segmentation-wise fitter for fitting of mean segmentation values."""

    def __init__(self, solver: CurveFitSolver, **fitter_kwargs):
        """Initialize with a solver.

        Args:
            solver: An instance of CurveFitSolver for optimization.
            **fitter_kwargs: Additional keyword arguments for fitter configuration.
        """
        super().__init__(solver=solver, **fitter_kwargs)
        self.segment_labels: np.ndarray | None = None
        self.segment_positions: list | None = None

    def fit(
        self,
        xdata: np.ndarray,
        image: np.ndarray,
        segmentation: np.ndarray | None = None,
        fixed_param_maps: dict[str, np.ndarray] | None = None,
        **fit_kwargs,
    ) -> "SegmentationWiseFitter":
        """Fit the model to mean values of each segmented region.

        Args:
            xdata: 1D array of independent variable (e.g., b-values).
            image: 2D, 3D or 4D array of shape (X, Y, Z, N) where N is the number of measurements (e.g., b-values).
            segmentation: 1D, 2D, 3D array of shape (X, Y, Z) with integer labels for segmented regions. Fitting will be performed separately for each segment.
            fixed_param_maps: Optional per-segment fixed parameter maps.
                Each value must be a 3-D spatial array matching the image
                spatial shape ``(X, Y, Z)``.  The mean across each segment
                is computed and forwarded to the solver as
                ``pixel_fixed_params``.
            **fit_kwargs: Additional keyword arguments for fitting.
                bounds: Optional tuple of (lower_bounds, upper_bounds) for parameters. Each should be an array of shape (n_segments, n_params). If not provided, defaults to model bounds.
                initial_guess: Optional array of shape (n_segments, n_params) for initial parameter guesses. If not provided, defaults to model initial guess.

        Returns:
            self: Returns the fitted instance for chaining.
        """
        if segmentation is None:
            raise ValueError("segmentation is required for segmentation-wise fitting")
        # Input validation
        validate_xdata(xdata)
        validate_data_shapes(xdata, image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape
        segmentation = validate_segmentation(segmentation, image.shape)
        logger.debug(
            f"Fitting SegmentationWiseFitter with image shape {image.shape} and "
            f"segmentation shape {segmentation.shape}"
        )
        segs_to_fit = self._extract_segmentation_mean_signals(image, segmentation)

        # Compute per-segment mean for fixed param maps
        pixel_fixed_params: dict[str, np.ndarray] | None = None
        if fixed_param_maps is not None:
            validate_fixed_param_maps(
                fixed_param_maps,
                image.shape[:-1],
                self.solver.model._all_param_names,
            )
            pixel_fixed_params = {}
            for name, vol in fixed_param_maps.items():
                means = np.array(
                    [np.mean(vol[segmentation == seg]) for seg in self.segment_labels]
                )
                pixel_fixed_params[name] = means

        # Fit the model to the mean signals of each segment
        self.solver.fit(
            xdata,
            segs_to_fit,
            pixel_fixed_params=pixel_fixed_params,
            **fit_kwargs,
        )
        for param, values in self.solver.params_.items():
            self.fitted_params_[param] = values  # shape (n_segments, n_params)

        return self

    def _extract_segmentation_mean_signals(
        self, image: np.ndarray, segmentation: np.ndarray
    ) -> np.ndarray:
        """Calculate mean signal for each segmented region."""
        unique_segments = np.unique(segmentation)
        mean_signals = []
        segment_positions = []
        for seg in unique_segments:
            mask = segmentation == seg
            mean_signal = np.mean(image[mask], axis=0)  # Mean across pixels in segment
            mean_signals.append(mean_signal)
            coords = np.column_stack(np.where(mask))  # shape (n_pixels_in_seg, ndim)
            segment_positions.append(coords)
        self.segment_labels = unique_segments
        self.segment_positions = (
            segment_positions  # List of arrays with pixel coordinates for each segment
        )
        return np.array(mean_signals)  # shape (n_segments, n_measurements)

    def predict(self, xdata: np.ndarray, **predict_kwargs) -> np.ndarray:
        """Predict the signal for each segmented region using the fitted parameters."""

        self._check_fitted()
        if xdata.ndim != 1:
            raise ValueError(
                f"Expected xdata to be 1D array, but got shape {xdata.shape}"
            )
        n_measurements = xdata.size

        # Collect params in model param order: shape (n_params, n_segments)
        param_names = self.solver.model.param_names
        popt_list = [self.fitted_params_[param] for param in param_names]
        popt = np.stack(popt_list, axis=0)  # shape (n_params, n_segments)

        n_segments = popt.shape[1]
        predictions = np.empty((n_segments, n_measurements), dtype=np.float64)
        for i in range(n_segments):
            segment_params = popt[:, i]
            predictions[i] = self.solver.model.forward(xdata, *segment_params)

        return predictions
