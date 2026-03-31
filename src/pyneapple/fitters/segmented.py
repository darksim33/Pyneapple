"""Segmented two-step fitter for sequential model fitting.

Orchestrates a two-step fitting pipeline:

1. **Step 1** — Fit a simple model (e.g. :class:`MonoExpModel`) using all or a
   subset of b-values to estimate baseline parameters such as ADC or T1.
2. **Step 2** — Fit a more complex model (e.g. :class:`BiExpModel`) using the
   full b-value set, optionally fixing parameters obtained from Step 1.

Example
-------
>>> from pyneapple.models import MonoExpModel, BiExpModel
>>> from pyneapple.solvers import CurveFitSolver
>>> from pyneapple.fitters import SegmentedFitter
>>>
>>> solver1 = CurveFitSolver(
...     model=MonoExpModel(), max_iter=200, tol=1e-8,
...     p0={"S0": 1.0, "D": 0.001},
...     bounds={"S0": (0, 5), "D": (0, 0.01)},
... )
>>> solver2 = CurveFitSolver(
...     model=BiExpModel(fit_reduced=True), max_iter=500, tol=1e-8,
...     p0={"f1": 0.2, "D1": 0.01, "D2": 0.001},
...     bounds={"f1": (0, 1), "D1": (0.001, 0.1), "D2": (0, 0.005)},
... )
>>> fitter = SegmentedFitter(
...     step1_solver=solver1,
...     step2_solver=solver2,
...     step1_bvalue_range=(200, None),
...     fixed_from_step1=["D"],
...     param_mapping={"D": "D2"},
... )
>>> fitter.fit(bvalues, image_4d, segmentation=mask_3d)
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from typing import Any

from ..solvers.base import BaseSolver
from ..utility.validation import (
    validate_xdata,
    validate_data_shapes,
    validate_segmentation,
)
from .base import BaseFitter
from .pixelwise import PixelWiseFitter


class SegmentedFitter(BaseFitter):
    """Two-step sequential model fitter.

    Fits a simple model first (e.g. monoexponential for ADC), then uses those
    results — optionally as fixed parameters — to initialise a second, more
    complex model fit (e.g. biexponential IVIM).

    Args:
        step1_solver: Solver for the first fitting step (e.g. MonoExpModel).
        step2_solver: Solver for the second fitting step (e.g. BiExpModel).
        step1_bvalue_range: Optional ``(lo, hi)`` range to select a b-value
            subset for Step 1.  Use ``None`` for an open end, e.g.
            ``(200, None)`` keeps all b >= 200.  If the entire tuple is
            ``None``, all b-values are used.
        fixed_from_step1: Parameter names from Step 1 to fix in Step 2.
            E.g. ``["D"]`` fixes the ADC as a per-pixel constant in the
            second step.
        param_mapping: Maps Step 1 parameter names to Step 2 parameter names.
            E.g. ``{"D": "D2"}`` maps MonoExp ``D`` to BiExp ``D2``.
            Names not present in the mapping use identity (same name).
        **fitter_kwargs: Additional keyword arguments for fitter configuration.
    """

    def __init__(
        self,
        step1_solver: BaseSolver,
        step2_solver: BaseSolver,
        step1_bvalue_range: tuple[float | None, float | None] | None = None,
        fixed_from_step1: list[str] | None = None,
        param_mapping: dict[str, str] | None = None,
        **fitter_kwargs,
    ):
        super().__init__(solver=step2_solver, **fitter_kwargs)
        self.step1_solver = step1_solver
        self.step2_solver = step2_solver
        self.step1_bvalue_range = step1_bvalue_range
        self.fixed_from_step1 = fixed_from_step1 or []
        self.param_mapping = param_mapping or {}

        self.step1_params_: dict[str, np.ndarray] = {}

        self._validate_init()

    def _validate_init(self) -> None:
        """Validate constructor arguments."""
        # Ensure fixed_from_step1 names are valid Step 1 model params
        step1_all = self.step1_solver.model._all_param_names
        for name in self.fixed_from_step1:
            if name not in step1_all:
                raise ValueError(
                    f"fixed_from_step1 name {name!r} is not a parameter of the "
                    f"Step 1 model. Available: {step1_all}"
                )

        # Ensure mapped target names are valid Step 2 model params
        step2_all = self.step2_solver.model._all_param_names
        for src in self.fixed_from_step1:
            dst = self.param_mapping.get(src, src)
            if dst not in step2_all:
                raise ValueError(
                    f"Mapped parameter {dst!r} (from {src!r}) is not a parameter "
                    f"of the Step 2 model. Available: {step2_all}"
                )

    def _subset_bvalues(
        self, xdata: np.ndarray, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select a b-value subset for Step 1.

        Args:
            xdata: Full b-value array.
            image: Full image array with last dimension matching xdata.

        Returns:
            Tuple of (subset_xdata, subset_image).
        """
        if self.step1_bvalue_range is None:
            return xdata, image

        lo, hi = self.step1_bvalue_range
        mask = np.ones(len(xdata), dtype=bool)
        if lo is not None:
            mask &= xdata >= lo
        if hi is not None:
            mask &= xdata <= hi

        if not np.any(mask):
            raise ValueError(
                f"No b-values fall within the range {self.step1_bvalue_range}. "
                f"Available b-values: {xdata}"
            )

        n_selected = int(np.sum(mask))
        if n_selected < 3:
            raise ValueError(
                f"Step 1 requires at least 3 b-values for fitting, but only "
                f"{n_selected} fall within the range {self.step1_bvalue_range}."
            )

        return xdata[mask], image[..., mask]

    def fit(
        self,
        xdata: np.ndarray,
        image: np.ndarray,
        segmentation: np.ndarray | None = None,
        **fit_kwargs,
    ) -> "SegmentedFitter":
        """Fit using the two-step segmented approach.

        Args:
            xdata: 1D array of b-values.
            image: 4D array of shape ``(X, Y, Z, N)``.
            segmentation: Optional binary mask of shape ``(X, Y, Z)``.
            **fit_kwargs: Additional keyword arguments forwarded to both
                fitting steps.

        Returns:
            self: Returns the fitted instance for chaining.
        """
        validate_xdata(xdata)
        validate_data_shapes(xdata, image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape

        if segmentation is not None:
            segmentation = validate_segmentation(segmentation, image.shape)

        # --- Step 1: Fit simple model (e.g. MonoExp) ---
        step1_xdata, step1_image = self._subset_bvalues(xdata, image)

        logger.debug(
            f"SegmentedFitter Step 1: {self.step1_solver.model.__class__.__name__} "
            f"with {len(step1_xdata)} b-values"
        )

        step1_fitter = PixelWiseFitter(solver=self.step1_solver)
        step1_fitter.fit(step1_xdata, step1_image, segmentation, **fit_kwargs)
        self.step1_params_ = dict(step1_fitter.get_fitted_params())

        # --- Build fixed_param_maps for Step 2 ---
        fixed_param_maps: dict[str, np.ndarray] | None = None
        if self.fixed_from_step1:
            fixed_param_maps = {}
            for src_name in self.fixed_from_step1:
                dst_name = self.param_mapping.get(src_name, src_name)
                vol = self._reconstruct_volume(
                    self.step1_params_[src_name],
                    step1_fitter.pixel_indices,
                    image.shape[:3],
                )
                fixed_param_maps[dst_name] = vol

        # --- Step 2: Fit complex model (e.g. BiExp) ---
        logger.debug(
            f"SegmentedFitter Step 2: {self.step2_solver.model.__class__.__name__} "
            f"with {len(xdata)} b-values"
            + (f", fixing {list(fixed_param_maps)}" if fixed_param_maps else "")
        )

        step2_fitter = PixelWiseFitter(solver=self.step2_solver)
        step2_fitter.fit(
            xdata,
            image,
            segmentation,
            fixed_param_maps=fixed_param_maps,
            **fit_kwargs,
        )

        # --- Store combined results ---
        self.fitted_params_ = dict(step2_fitter.get_fitted_params())
        self.pixel_indices = step2_fitter.pixel_indices

        # Merge fixed params back into fitted_params_ for completeness
        for src_name in self.fixed_from_step1:
            dst_name = self.param_mapping.get(src_name, src_name)
            self.fitted_params_[dst_name] = self.step1_params_[src_name]

        return self

    def predict(
        self, xdata: np.ndarray[tuple[Any, ...], np.dtype[Any]], **predict_kwargs
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        """Predict the signal for each pixel using the fitted parameters."""
        return super().predict(xdata, **predict_kwargs)

    def _get_param_names(self):
        """Get the full set of parameter names in the order expected by the Step 2 model."""
        return self.solver.model._all_param_names
