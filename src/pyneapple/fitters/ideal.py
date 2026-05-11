"""IDEAL fitter for independent fitting of each pixel."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from loguru import logger

from ..solvers import CurveFitSolver
from ..utility.validation import (
    validate_data_shapes,
    validate_parameter_names,
    validate_segmentation,
    validate_xdata,
)
from .base import BaseFitter

_DOWNSAMPLING_METHODS = ["linear", "cubic", "area", "block_average"]
_UPSAMPLING_METHODS = ["linear", "cubic"]
_BLOCK_AVERAGE = "block_average"


class IDEALFitter(BaseFitter):
    """IDEAL fitter for independent fitting of each pixel."""

    def __init__(
        self,
        solver: CurveFitSolver,
        dim_steps: np.ndarray,
        step_tol: dict[str, float],
        ideal_dims: int = 2,
        segmentation_threshold: float = 0.025,
        downsampling_method: str = "block_average",
        upsampling_method: str = "cubic",
        clamp_interpolated_p0: bool = False,
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
            step_tol: Dict mapping each model parameter name to its tolerance
                fraction. Used to scale bounds around the interpolated parameter
                map at each IDEAL step after the first. Keys must match
                ``solver.model.param_names``. Example:
                ``{"S0": 0.5, "f1": 0.2, "D1": 0.2, "D2": 0.2}``.
            segmentation_threshold: Threshold for including pixels in fitting
                based on segmentation. Default is 0.2 (20% of maximum).
            downsampling_method: Method for downsampling the signal image and
                segmentation at each IDEAL step. Must be one of
                ``"linear"``, ``"cubic"``, ``"area"``, or
                ``"block_average"``. Default is ``"block_average"``.
                ``"block_average"`` uses NaN-masked bin-mean averaging
                (equivalent to the MATLAB ``accumarray(@nanmean)`` approach),
                where pixels outside the segmentation are excluded from each
                spatial bin. The remaining methods delegate to ``cv2.resize``.
            upsampling_method: Method for upsampling the parameter map from one
                IDEAL step to the next. Must be one of ``"linear"`` or
                ``"cubic"``. Default is ``"cubic"``. ``"block_average"`` and
                ``"area"`` are not valid for upsampling.
            clamp_interpolated_p0: If ``True`` (default), the upsampled parameter
                map is clamped to the global solver bounds before step-wise
                tolerance bounds are derived. Set to ``False`` to allow
                interpolated values outside the solver bounds to be used as
                initial guesses directly (the step bounds themselves are still
                clipped to the solver bounds).
            **fitter_kwargs: Additional keyword arguments for fitter configuration.
        """
        super().__init__(solver=solver, **fitter_kwargs)
        self.dim_steps = dim_steps
        self.step_tol = step_tol
        self.ideal_dims = ideal_dims
        self.segmentation_threshold = segmentation_threshold
        self.downsampling_method = self._get_downsampling_method(downsampling_method)
        self.upsampling_method = self._get_upsampling_method(upsampling_method)
        self.clamp_interpolated_p0 = clamp_interpolated_p0
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

    def _get_downsampling_method(self, method: str) -> int | str:
        """Validate and return the downsampling method.

        Args:
            method: One of ``"linear"``, ``"cubic"``, ``"area"``, or
                ``"block_average"``.

        Returns:
            The ``cv2`` interpolation flag for cv2-based methods, or the
            sentinel string ``"block_average"`` for the NaN-masked bin-mean
            method.

        Raises:
            ValueError: If ``method`` is not in ``_DOWNSAMPLING_METHODS``.
        """
        if method not in _DOWNSAMPLING_METHODS:
            raise ValueError(
                f"Invalid downsampling method: {method!r}. "
                f"Must be one of {_DOWNSAMPLING_METHODS}."
            )
        if method == "linear":
            return cv2.INTER_LINEAR
        elif method == "cubic":
            return cv2.INTER_CUBIC
        elif method == "area":
            return cv2.INTER_AREA
        else:  # "block_average"
            return _BLOCK_AVERAGE

    def _get_upsampling_method(self, method: str) -> int:
        """Validate and return the upsampling method.

        Args:
            method: One of ``"linear"`` or ``"cubic"``.  ``"block_average"``
                and ``"area"`` are not meaningful for upsampling and are
                rejected.

        Returns:
            The ``cv2`` interpolation flag.

        Raises:
            ValueError: If ``method`` is not in ``_UPSAMPLING_METHODS``.
        """
        if method not in _UPSAMPLING_METHODS:
            raise ValueError(
                f"Invalid upsampling method: {method!r}. "
                f"Must be one of {_UPSAMPLING_METHODS}."
            )
        if method == "linear":
            return cv2.INTER_LINEAR
        else:  # "cubic"
            return cv2.INTER_CUBIC

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
        self._validate_solver_bounds()
        self._validate_fitter_inputs(self.dim_steps, self.ideal_dims)
        image = self._validate_image_dims(image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape

        _t0 = time.perf_counter()
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
        # Expand segmentation to 4D so downsampling helpers can process it uniformly
        if segmentation.ndim == 3:
            segmentation = segmentation[..., np.newaxis]

        logger.debug(
            f"Fitting IDEALFitter with image shape {image.shape} and fitting "
            f"{self.n_measurements} measurements."
        )

        param_names = self.solver.model.param_names
        n_params = len(param_names)
        self.step_params = []  # reset for potential re-fitting

        # Global solver bounds — used to initialise step 0 and to clamp
        # interpolated p0/lb/ub at every subsequent step so that cubic
        # interpolation overshoot never produces out-of-range or negative values.
        p0_vals = np.array([self.solver.p0[n] for n in param_names])
        lo_vals = np.array([self.solver.bounds[n][0] for n in param_names])
        hi_vals = np.array([self.solver.bounds[n][1] for n in param_names])

        # --- Resampling of the image to the IDEAL grid

        for step_index, step in enumerate(dim_steps):
            step_shape = tuple(int(s) for s in step)
            logger.debug(f"Starting IDEAL step {step_index} with dim_steps={step}")
            if step_index == 0:
                p0 = np.broadcast_to(p0_vals, (*step_shape, n_params)).copy()
                lower_bounds = np.broadcast_to(lo_vals, (*step_shape, n_params)).copy()
                upper_bounds = np.broadcast_to(hi_vals, (*step_shape, n_params)).copy()
            else:
                prev_param_map = self.step_params[-1]  # shape (*prev_shape, n_params)
                # Parameter map is always upsampled (never block-averaged).
                p0 = self._upsampling_array(prev_param_map, step_shape)
                # Cubic interpolation can overshoot and produce values outside
                # the original parameter range (including negatives).  Clamp p0
                # to the global solver bounds before deriving step bounds
                # unless the user has opted out.
                if self.clamp_interpolated_p0:
                    p0 = np.clip(p0, lo_vals, hi_vals)
                tol_vals = np.array([self.step_tol[n] for n in param_names])
                lower_bounds = np.clip(p0 * (1 - tol_vals), lo_vals, hi_vals)
                upper_bounds = np.clip(p0 * (1 + tol_vals), lo_vals, hi_vals)

            # Image — downsample; when block_average, pass full-res binary mask
            # so that out-of-ROI pixels are excluded from each spatial bin mean.
            if self.downsampling_method == _BLOCK_AVERAGE:
                _orig_mask = segmentation[..., 0] > self.segmentation_threshold
                _image = self._downsampling_array(image, step_shape, mask=_orig_mask)
            else:
                _image = self._downsampling_array(image, step_shape)

            # Segmentation — downsample without mask; fractional bin-fraction
            # values are then thresholded to produce the step binary mask.
            _segmentation_interp = self._downsampling_array(segmentation, step_shape)

            # Squeeze last dim to get 3D bool mask compatible with _extract_pixel_data
            _segmentation_mask = (
                _segmentation_interp[..., 0] > self.segmentation_threshold
            )
            # Fallback: if the ROI is too sparse to survive downsampling at this
            # resolution (e.g. kidney in a large FOV at a 2×2 grid step), fit all
            # voxels so the multi-resolution chain does not collapse to 0 pixels.
            if not _segmentation_mask.any():
                logger.warning(
                    "IDEAL step {}: 0 voxels above segmentation_threshold {:.3f} "
                    "at resolution {} — falling back to all voxels for this step.",
                    step_index,
                    self.segmentation_threshold,
                    step_shape,
                )
                _segmentation_mask = np.ones(step_shape, dtype=bool)

            pixel_to_fit = self._extract_pixel_data(_image, _segmentation_mask)
            pixel_positions = list(self.pixel_indices)  # save before any overwrite
            # (n_params, n_pixels) as required by CurveFitSolver
            p0_to_fit = p0[_segmentation_mask].T  # shape (n_params, n_pixels)
            lower_to_fit = lower_bounds[
                _segmentation_mask
            ].T  # shape (n_params, n_pixels)
            upper_to_fit = upper_bounds[
                _segmentation_mask
            ].T  # shape (n_params, n_pixels)
            bounds_to_fit = (lower_to_fit, upper_to_fit)

            logger.debug(
                "IDEAL step {} | resolution={} | n_pixels={}",
                step_index,
                step_shape,
                p0_to_fit.shape[1] if p0_to_fit.ndim == 2 else 0,
            )
            for p_idx, p_name in enumerate(param_names):
                row = p0_to_fit[p_idx] if p0_to_fit.ndim == 2 else np.array([])
                lb_row = lower_to_fit[p_idx] if lower_to_fit.ndim == 2 else np.array([])
                ub_row = upper_to_fit[p_idx] if upper_to_fit.ndim == 2 else np.array([])
                logger.debug(
                    "  {:>6s}  p0=[{:.4g}, {:.4g}]  lb=[{:.4g}, {:.4g}]  ub=[{:.4g}, {:.4g}]",
                    p_name,
                    float(row.min()) if row.size else float("nan"),
                    float(row.max()) if row.size else float("nan"),
                    float(lb_row.min()) if lb_row.size else float("nan"),
                    float(lb_row.max()) if lb_row.size else float("nan"),
                    float(ub_row.min()) if ub_row.size else float("nan"),
                    float(ub_row.max()) if ub_row.size else float("nan"),
                )

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

        fit_time = time.perf_counter() - _t0
        # pixel_to_fit corresponds to the final IDEAL step (full resolution)
        self.results_ = self._assemble_fit_result(xdata, pixel_to_fit, fit_time)

        return self

    def _validate_step_tol(self):
        """Validate that step_tol is a dict whose keys match model.param_names."""
        if not isinstance(self.step_tol, dict):
            raise ValueError(
                "step_tol must be a dict mapping parameter names to tolerance "
                f"fractions, e.g. {{'S0': 0.5, 'D': 0.2}}. "
                f"Got {type(self.step_tol).__name__}."
            )
        try:
            validate_parameter_names(self.step_tol, self.solver.model.param_names)
        except ValueError as exc:
            raise ValueError(
                f"step_tol keys {set(self.step_tol.keys())} do not match model "
                f"parameter names {self.solver.model.param_names}: {exc}"
            ) from exc

    def _validate_solver_bounds(self):
        """Validate that no lower solver bound is zero.

        IDEAL uses multiplicative bound scaling at each resolution step:

            lower = p0 * (1 - tol)
            upper = p0 * (1 + tol)

        A lower bound of zero allows interpolated p0 values to reach zero,
        which collapses both step bounds to zero (lb == ub) and causes
        scipy curve_fit to raise an ill-formed bounds error.  All lower
        bounds must be strictly greater than zero.
        """
        zero_params = [
            name
            for name in self.solver.model.param_names
            if self.solver.bounds[name][0] <= 0
        ]
        if zero_params:
            raise ValueError(
                "IDEAL fitting requires all lower bounds to be strictly greater "
                "than zero.  The following parameters have a lower bound of zero "
                f"or less: {zero_params}.  "
                "Set a small positive lower bound (e.g. S0 = [1.0, 5000.0]) to "
                "prevent step bounds from collapsing during multi-resolution fitting."
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

    # ------------------------------------------------------------------
    # Resampling helpers
    # ------------------------------------------------------------------

    def _downsampling_array(
        self,
        array: np.ndarray,
        target_shape: tuple[int, int, int],
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Downsample a 4D array to ``target_shape``.

        Dispatches to :meth:`_block_average_array` when
        ``self.downsampling_method == "block_average"``, otherwise delegates
        to ``cv2.resize`` with ``self.downsampling_method`` as the
        interpolation flag.

        Args:
            array: 4-D input array of shape ``(X, Y, Z, C)``.
            target_shape: Desired spatial shape ``(X, Y, Z)``.
            mask: Optional 3-D boolean array of shape ``(X, Y, Z)``.  When
                provided (and ``downsampling_method == "block_average"``),
                pixels where ``mask`` is ``False`` are set to ``NaN`` before
                binning so they do not contribute to the bin mean.  Ignored
                for cv2-based methods.

        Returns:
            Resampled array of shape ``(*target_shape, C)``.
        """
        if self.downsampling_method == _BLOCK_AVERAGE:
            return self._block_average_array(array, target_shape, mask=mask)

        target_shape = tuple(int(s) for s in target_shape)
        if array.dtype.kind not in ("f",):
            array = array.astype(np.float32)
        result = np.zeros((*target_shape, array.shape[-1]), dtype=array.dtype)
        for nslice in range(array.shape[-2]):
            for i in range(array.shape[-1]):
                result[..., nslice, i] = cv2.resize(
                    array[..., nslice, i],
                    (target_shape[1], target_shape[0]),
                    interpolation=self.downsampling_method,
                )
        return result

    def _upsampling_array(
        self,
        array: np.ndarray,
        target_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Upsample a 4D array to ``target_shape`` using ``cv2.resize``.

        Always uses ``self.upsampling_method`` (a ``cv2`` interpolation flag).
        Block averaging is not meaningful for upsampling and is therefore not
        available here.

        Args:
            array: 4-D input array of shape ``(X, Y, Z, C)``.
            target_shape: Desired spatial shape ``(X, Y, Z)``.

        Returns:
            Resampled array of shape ``(*target_shape, C)``.
        """
        target_shape = tuple(int(s) for s in target_shape)
        if array.dtype.kind not in ("f",):
            array = array.astype(np.float32)
        result = np.zeros((*target_shape, array.shape[-1]), dtype=array.dtype)
        for nslice in range(array.shape[-2]):
            for i in range(array.shape[-1]):
                result[..., nslice, i] = cv2.resize(
                    array[..., nslice, i],
                    (target_shape[1], target_shape[0]),
                    interpolation=self.upsampling_method,
                )
        return result

    def _block_average_array(
        self,
        array: np.ndarray,
        target_shape: tuple[int, int, int],
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Downsample a 4D array by NaN-masked spatial bin averaging.

        This is a Python port of the MATLAB ``accumarray(@nanmean)`` approach:

        .. code-block:: matlab

            dx = size(data, 1) / steps(1);
            [r, c] = ndgrid(1:size(data_masked,1), 1:size(data_masked,2));
            [~, ibin] = histc(r(:), 0.5:dx:size(data_masked,1)+0.5);
            [~, jbin] = histc(c(:), 0.5:dx:size(data_masked,2)+0.5);
            data_downscaled = accumarray(idx, data_masked(:), [nr*nc 1], @nanmean);

        Each input pixel ``r`` (0-indexed) is assigned to bin
        ``clip(floor((r + 0.5) / dx), 0, target - 1)``, which is the exact
        Python equivalent of MATLAB's ``histc`` with edges
        ``0.5 : dx : N + 0.5``.

        Args:
            array: 4-D input array of shape ``(X, Y, Z, C)``.
            target_shape: Desired spatial shape ``(tx, ty, tz)``.  The z
                dimension is preserved unchanged (IDEAL only steps in X/Y).
            mask: Optional 3-D boolean array of shape ``(X, Y, Z)``.  Where
                ``False``, pixels are treated as ``NaN`` and excluded from
                their bin's mean.  Bins with no valid pixels default to
                ``0.0``.

        Returns:
            Resampled array of shape ``(*target_shape, C)`` with dtype
            ``float64``.
        """
        target_shape = tuple(int(s) for s in target_shape)
        orig_x, orig_y, orig_z, n_ch = array.shape
        target_x, target_y, target_z = target_shape

        output = np.zeros((*target_shape, n_ch), dtype=np.float64)

        # Bin assignment for rows and columns (0-indexed pixels).
        # floor((r + 0.5) / dx) is the Python equivalent of MATLAB histc with
        # edges 0.5:dx:N+0.5 applied to 1-indexed pixels r = 1..N.
        dx = orig_x / target_x
        dy = orig_y / target_y
        r_bins = np.clip(
            np.floor((np.arange(orig_x) + 0.5) / dx).astype(int), 0, target_x - 1
        )
        c_bins = np.clip(
            np.floor((np.arange(orig_y) + 0.5) / dy).astype(int), 0, target_y - 1
        )

        # Pre-compute the flat linear bin index for every input pixel.
        # Shape: (orig_x, orig_y) → ravelled to (orig_x * orig_y,)
        bin_idx_flat = (
            r_bins[:, np.newaxis] * target_y + c_bins[np.newaxis, :]
        ).ravel()
        n_bins = target_x * target_y

        for z in range(orig_z):
            for i in range(n_ch):
                temp = array[:, :, z, i].astype(np.float64).ravel()

                if mask is not None:
                    # Pixels outside the mask are excluded from bin means.
                    out_of_mask = ~mask[:, :, z].ravel()
                    temp = temp.copy()
                    temp[out_of_mask] = np.nan

                not_nan = ~np.isnan(temp)
                if not_nan.any():
                    valid_idx = bin_idx_flat[not_nan]
                    valid_vals = temp[not_nan]
                    count = np.bincount(valid_idx, minlength=n_bins)
                    total = np.bincount(valid_idx, weights=valid_vals, minlength=n_bins)
                    with np.errstate(invalid="ignore"):
                        result_flat = np.where(count > 0, total / count, 0.0)
                else:
                    result_flat = np.zeros(n_bins)

                output[:, :, z, i] = result_flat.reshape(target_x, target_y)

        return output

    def predict(
        self, xdata: np.ndarray[tuple[Any, ...], np.dtype[Any]], **predict_kwargs
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        """Predict the signal for each pixel using the fitted parameters."""
        return super().predict(xdata, **predict_kwargs)
