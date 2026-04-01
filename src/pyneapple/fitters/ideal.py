"""IDEAL fitter for independent fitting of each pixel."""

from typing import Any

import cv2
import numpy as np
from loguru import logger

from ..solvers import CurveFitSolver
from ..utility.validation import (
    validate_data_shapes,
    validate_segmentation,
    validate_xdata,
)
from .base import BaseFitter

_INTERPOLATION_METHODS = ["linear", "cubic"]


class IDEALFitter(BaseFitter):
    """IDEAL fitter for independent fitting of each pixel."""

    def __init__(
        self,
        solver: CurveFitSolver,
        dim_steps: np.ndarray,
        step_tol: list[float] | np.ndarray,
        ideal_dims: int = 2,
        segmentation_threshold: float = 0.2,
        interpolation_method: str = "cubic",
        **fitter_kwargs,
    ):
        """Initialize the IDEAL fitter.

        Args:
            solver: An instance of CurveFitSolver for optimization.
            dim_steps: 2D array of shape (n_steps, ideal_dims) specifying the
                grid resolutions at each IDEAL step. Each row is one step,
                e.g. [[16, 16], [32, 32], [64, 64], [128, 128]].
            ideal_dims: Number of dimensions in the IDEAL grid (e.g. 2 for
                2D grid). Default is 2.
            step_tol: List of tolerances for each parameter to determine
                boundaries for each IDEAL step. Length must match the number
                of model parameters.
            segmentation_threshold: Threshold for including pixels in fitting
                based on segmentation. Default is 0.2 (20% of maximum).
            interpolation_method: Method for interpolating the IDEAL grid.
                Must be one of ``"linear"`` or ``"cubic"``. Default is
                ``"cubic"``.
            **fitter_kwargs: Additional keyword arguments for fitter configuration.
        """
        super().__init__(solver=solver, **fitter_kwargs)
        self.dim_steps = dim_steps
        self.step_tol = step_tol
        self.ideal_dims = ideal_dims
        self.segmentation_threshold = segmentation_threshold
        self.interpolation_method = self._get_interpolation_method(interpolation_method)
        self.step_params: list[np.ndarray] = []  # To store parameter maps for each step

    def _validate_fitter_inputs(self, dim_steps: np.ndarray, ideal_dims: int):
        """Validate inputs for fitting."""
        if dim_steps.ndim != 2:
            raise ValueError(
                "dim_steps must be a 2D array of shape (n_steps, ideal_dims)."
            )
        if dim_steps.shape[1] != ideal_dims:
            raise ValueError(
                f"dim_steps must have {ideal_dims} columns corresponding to ideal_dims."
            )
        # Check dim_steps increase monotonically along each step
        for i in range(dim_steps.shape[0] - 1):
            if not np.all(dim_steps[i + 1] > dim_steps[i]):
                raise ValueError(
                    f"dim_steps row {i + 1} must be greater than row {i} (monotonic increase)."
                )

    def _get_interpolation_method(self, method: str):
        """Get the interpolation method for IDEAL fitting."""
        if method not in _INTERPOLATION_METHODS:
            raise ValueError(
                f"Invalid interpolation method: {method}. Must be one of {_INTERPOLATION_METHODS}."
            )
        if method == "linear":
            return cv2.INTER_LINEAR
        elif method == "cubic":
            return cv2.INTER_CUBIC
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    def fit(
        self,
        xdata: np.ndarray,
        image: np.ndarray,
        segmentation: np.ndarray | None = None,
        **fit_kwargs,
    ) -> "IDEALFitter":
        """Fit the model to each pixel independently.

        Args:
            xdata: 1D array of independent variable (e.g., b-values).
            image: 2D, 3D or 4D array of shape (X, Y, Z, N) where N is the number of measurements (e.g., b-values).
            segmentation: Optional 1D, 2D, 3D array of shape (X, Y, Z) with integer labels for segmented regions. If provided, fitting will be performed separately for each segment.
            **fit_kwargs: Additional keyword arguments for fitting.
                bounds: Optional tuple of (lower_bounds, upper_bounds) for parameters. Each should be an array of shape (n_pixels, n_params). If not provided, defaults to model bounds.
                initial_guess: Optional array of shape (n_pixels, n_params) for initial parameter guesses. If not provided, defaults to model initial guess.
        """

        # --- Input validation and setup
        validate_xdata(xdata)
        validate_data_shapes(xdata, image)
        self._validate_step_tol()
        self._validate_fitter_inputs(self.dim_steps, self.ideal_dims)
        image = self._validate_image_dims(image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape
        # Validate last step matches image spatial dimensions
        if not np.allclose(self.dim_steps[-1], image.shape[: self.ideal_dims]):
            raise ValueError(
                "The last step in dim_steps must match the spatial dimensions of the "
                "image."
            )
        # if dim_steps is for 2D but image is 4D (with slice dim), add z-dimension locally.
        # We use a local variable to avoid permanently mutating self.dim_steps, which
        # would break _validate_fitter_inputs on a second call to fit().
        if self.ideal_dims == 2 and image.ndim == 4:
            z_col = np.full((self.dim_steps.shape[0], 1), image.shape[2])
            dim_steps = np.hstack([self.dim_steps, z_col])
        else:
            dim_steps = self.dim_steps

        if segmentation is not None:
            segmentation = validate_segmentation(segmentation, image.shape)
        else:
            segmentation = np.ones(
                image.shape[:3], dtype=int
            )  # Select all pixels if no segmentation provided.
        # Expand segmentation to 4D so _interpolate_array can process it uniformly
        if segmentation.ndim == 3:
            segmentation = segmentation[..., np.newaxis]

        logger.debug(
            f"Fitting IDEALFitter with image shape {image.shape} and fitting "
            f"{self.n_measurements} measurements."
        )

        param_names = self.solver.model.param_names
        n_params = len(param_names)
        self.step_params = []  # reset for potential re-fitting

        # --- Interpolation of the image to the IDEAL grid

        for step_index, step in enumerate(dim_steps):
            step_shape = tuple(int(s) for s in step)
            logger.debug(f"Starting IDEAL step {step_index} with dim_steps={step}")
            if step_index == 0:
                p0_vals = np.array([self.solver.p0[n] for n in param_names])
                lo_vals = np.array([self.solver.bounds[n][0] for n in param_names])
                hi_vals = np.array([self.solver.bounds[n][1] for n in param_names])
                p0 = np.broadcast_to(p0_vals, (*step_shape, n_params)).copy()
                lower_bounds = np.broadcast_to(lo_vals, (*step_shape, n_params)).copy()
                upper_bounds = np.broadcast_to(hi_vals, (*step_shape, n_params)).copy()
            else:
                prev_param_map = self.step_params[-1]  # shape (*prev_shape, n_params)
                p0 = self._interpolate_array(
                    prev_param_map, step_shape
                )  # interpolate to current step
                lower_bounds = p0 * (
                    1 - np.array(self.step_tol)
                )  # set bounds based on step_tol
                upper_bounds = p0 * (1 + np.array(self.step_tol))
            _image = self._interpolate_array(
                image, step_shape
            )  # interpolate image to current step
            _segmentation_interp = self._interpolate_array(segmentation, step_shape)
            # Squeeze last dim to get 3D bool mask compatible with _extract_pixel_data
            _segmentation_mask = (
                _segmentation_interp[..., 0] > self.segmentation_threshold
            )

            pixel_to_fit = self._extract_pixel_data(_image, _segmentation_mask)
            pixel_positions = list(self.pixel_indices)  # save before any overwrite
            # (n_params, n_pixels) as required by CurveFitSolver
            p0_to_fit = p0[_segmentation_mask].T
            lower_to_fit = lower_bounds[_segmentation_mask].T
            upper_to_fit = upper_bounds[_segmentation_mask].T
            bounds_to_fit = (lower_to_fit, upper_to_fit)

            self.solver.fit(
                xdata, pixel_to_fit, p0=p0_to_fit, bounds=bounds_to_fit, **fit_kwargs
            )

            # Reconstruct spatial param map shape (*step_shape, n_params)
            # and append to step_params so the next iteration can interpolate from it
            param_map = np.zeros((*step_shape, n_params), dtype=np.float64)
            if pixel_positions:
                xs = [pos[0] for pos in pixel_positions]
                ys = [pos[1] for pos in pixel_positions]
                zs = [pos[2] for pos in pixel_positions]
                for param_idx, param_name in enumerate(param_names):
                    values = np.atleast_1d(self.solver.params_[param_name])
                    param_map[xs, ys, zs, param_idx] = values
            self.step_params.append(param_map)

        for param, values in self.solver.params_.items():
            self.fitted_params_[param] = values

        return self

    def _validate_step_tol(self):
        if len(self.step_tol) != self.solver.model.n_params:
            raise ValueError(
                f"step_tol must have length equal to the number of model parameters ({self.solver.model.n_params})."
            )

    def _validate_image_dims(self, image: np.ndarray) -> np.ndarray:
        """Validate image is a 4D array with shape (x,y,slice,measurement).

        Returns:
            np.ndarray: The image, expanded to 4D if it was 3D.
        """
        # Bug 2 fix: return the (possibly expanded) image so the caller sees the change.
        if image.ndim == 4:
            return image
        elif image.ndim == 3:
            if self.ideal_dims == 3:
                raise ValueError(
                    f"Image dimension ({image.ndim}) not sufficient for 3D interpolation (ideal_dims={self.ideal_dims})"
                )
            return np.expand_dims(image, axis=-2)
        else:
            raise ValueError(f"Image Array needs to be 3 or 4 not {image.ndim}")

    def _interpolate_array(
        self, array: np.ndarray, target_shape: tuple[int, int, int]
    ) -> np.ndarray:
        """Interpolate a 4D array to the target shape using the specified method."""
        # ensure target_shape is a plain Python tuple of ints so that
        # tuple concatenation works correctly (NumPy arrays use element-wise +).
        target_shape = tuple(int(s) for s in target_shape)
        # cv2.resize only supports float32, float64, uint8, uint16, int16.
        # Integer arrays (e.g., int64 segmentation masks) must be cast to float32 first
        # so that cv2 can process them and the interpolated values are usable for
        # subsequent threshold comparisons.
        if array.dtype.kind not in ("f",):
            array = array.astype(np.float32)
        interpolated = np.zeros((*target_shape, array.shape[-1]), dtype=array.dtype)
        for nslice in range(array.shape[-2]):
            for i in range(array.shape[-1]):
                interpolated[..., nslice, i] = cv2.resize(
                    array[..., nslice, i],
                    (target_shape[1], target_shape[0]),
                    interpolation=self.interpolation_method,
                )
        return interpolated

    def predict(
        self, xdata: np.ndarray[tuple[Any, ...], np.dtype[Any]], **predict_kwargs
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        """Predict the signal for each pixel using the fitted parameters."""
        return super().predict(xdata, **predict_kwargs)
