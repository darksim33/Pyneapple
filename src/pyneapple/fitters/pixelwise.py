"""Pixel-wise fitter for independent fitting of each pixel."""

from typing import Any
from loguru import logger
from tqdm import tqdm

import numpy as np

from .base import BaseFitter
from ..solvers import CurveFitSolver
from ..utility.validation import (
    validate_xdata,
    validate_data_shapes,
    validate_segmentation,
    validate_fixed_param_maps,
)


class PixelWiseFitter(BaseFitter):
    def __init__(self, solver: CurveFitSolver, **fitter_kwargs):
        """Pixel-wise fitter for independent fitting of each pixel.

        Args:
            solver: An instance of CurveFitSolver for optimization.
            **fitter_kwargs: Additional keyword arguments for fitter configuration.
        """
        super().__init__(solver=solver, **fitter_kwargs)

    def fit(
        self,
        xdata: np.ndarray,
        image: np.ndarray,
        segmentation: np.ndarray | None = None,
        fixed_param_maps: dict[str, np.ndarray] | None = None,
        **fit_kwargs,
    ) -> "PixelWiseFitter":
        """Fit the model to each pixel independently.

        Args:
            xdata: 1D array of independent variable (e.g., b-values).
            image: 2D, 3D or 4D array of shape (X, Y, Z, N) where N is the number of measurements (e.g., b-values).
            segmentation: Optional 1D, 2D, 3D array of shape (X, Y, Z) with integer labels for segmented regions. If provided, fitting will be performed separately for each segment.
            fixed_param_maps: Optional per-pixel fixed parameter maps.
                Each value must be a 3-D spatial array matching the image
                spatial shape ``(X, Y, Z)``.  The array is flattened to the
                same pixel ordering used internally and forwarded to the
                solver as ``pixel_fixed_params``.
            **fit_kwargs: Additional keyword arguments for fitting.
                bounds: Optional tuple of (lower_bounds, upper_bounds) for parameters. Each should be an array of shape (n_pixels, n_params). If not provided, defaults to model bounds.
                initial_guess: Optional array of shape (n_pixels, n_params) for initial parameter guesses. If not provided, defaults to model initial guess.

        Returns:
            self: Returns the fitted instance for chaining.
        """
        # Input validation
        validate_xdata(xdata)
        validate_data_shapes(xdata, image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape
        if segmentation is not None:
            segmentation = validate_segmentation(segmentation, image.shape)
        else:
            segmentation = np.ones(
                image.shape[:3], dtype=int
            )  # Select all pixels if no segmentation provided.
        logger.debug(
            f"Fitting PixelWiseFitter with image shape {image.shape} and "
            f"segmentation shape {segmentation.shape if segmentation is not None else 'None'}"
        )

        pixel_to_fit = self._extract_pixel_data(image, segmentation)

        # Flatten per-pixel fixed param maps to match pixel ordering
        pixel_fixed_params: dict[str, np.ndarray] | None = None
        if fixed_param_maps is not None:
            validate_fixed_param_maps(
                fixed_param_maps,
                image.shape[:-1],
                self.solver.model._all_param_names,
            )
            pixel_fixed_params = {}
            for name, vol in fixed_param_maps.items():
                pixel_fixed_params[name] = vol[segmentation != 0]

        self.solver.fit(
            xdata,
            pixel_to_fit,
            pixel_fixed_params=pixel_fixed_params,
            **fit_kwargs,
        )

        for param, values in self.solver.params_.items():
            self.fitted_params_[param] = values

        return self

    def predict(
        self, xdata: np.ndarray[tuple[Any, ...], np.dtype[Any]], **predict_kwargs
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        """Predict the signal for each pixel using the fitted parameters."""

        self._check_fitted()
        if xdata.ndim != 1:
            raise ValueError(
                f"Expected xdata to be 1D array, but got shape {xdata.shape}"
            )
        n_measurements = xdata.size

        # Collect params in model param oder: shape (n_parms, n_pixels)
        param_names = self.solver.model.param_names
        popt_list = [self.fitted_params_[param] for param in param_names]
        popt = np.stack(popt_list, axis=0)  # shape (n_params, n_pixels)

        n_pixels = popt.shape[1]
        predictions = np.empty((n_pixels, n_measurements), dtype=np.float64)
        for i in tqdm(range(n_pixels), total=n_pixels, desc="Predicting pixel-wise: "):
            pixel_params = popt[:, i]
            predictions[i] = self.solver.model.forward(xdata, *pixel_params)

        # Reconstruct image shape: (X, Y, Z, N)
        output_shape = self.image_shape[:-1] + (n_measurements,)
        pred_image = np.zeros(output_shape, dtype=np.float64)
        for idx, pixel in enumerate(self.pixel_indices):
            pred_image[pixel] = predictions[idx]

        return pred_image
