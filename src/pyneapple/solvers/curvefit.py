"""Scipy curve_fit based solver."""

from typing import Any
import numpy as np
from loguru import logger
from scipy.optimize import curve_fit
from tqdm import tqdm
from joblib import Parallel, delayed

from .base import BaseSolver
from ..utility import validation as validation_utils


class CurveFitSolver(BaseSolver):
    """Solver using scipy's curve_fit for non-linear least squares optimization.

    This solver is designed for fitting parametric models to diffusion MRI data
    using the Levenberg-Marquardt algorithm. It supports both single-voxel and
    multi-voxel fitting with optional progress bars.

    Args:
        model (BaseModel): Model instance with forward() and optionally jacobian() methods.
        max_iter (int): Maximum number of function evaluations (default 250).
        tol (float): Tolerance for convergence (default 1e-8).
        p0 (dict[str, np.ndarray]): Initial guess for parameters (default None, which uses model defaults).
        bounds (dict[str, tuple[np.ndarray, np.ndarray]]): Bounds on parameters as a tuple (lower, upper) (default None, which uses model defaults).
        verbose (bool, optional): If True, prints detailed fitting information (default False).
        method (str, optional): Optimization method to use (default 'lm' for Levenberg-Marquardt). Note: 'lm' does not support bounds.
        multi_threading (bool, optional): Enable multi-threading for multi-voxel fitting (default False). Uses joblib if True.
        use_jacobian (bool, optional): Use model's jacobian() method if available for faster convergence (default True).
        **solver_kwargs: Additional keyword arguments for curve_fit (e.g., sigma, absolute_sigma).
    """

    def __init__(
        self,
        model: Any,
        max_iter: int,
        tol: float,
        p0: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        verbose: bool = False,
        method: str = "trf",
        multi_threading: bool = False,
        use_jacobian: bool = True,
        **solver_kwargs,
    ):
        """Initialize CurveFitSolver with configuration.

        Args:
            model (BaseModel): Model instance with forward() method
            max_iter (int): Maximum function evaluations (default: 250)
            tol (float): Tolerance for termination (default: 1e-8)
            p0 (dict[str, float]): Initial parameter guesses (default: None, uses model defaults)
            bounds (dict[str, tuple[float, float]]): Parameter bounds as (lower, upper) (default: None, uses model defaults)
            verbose (bool): Enable verbose output (default: False)
            method (str): Scipy optimization method - 'trf', 'dogbox', 'lm' (default: 'trf')
            multi_threading (bool): Enable joblib multiprocessing (default: False)
            use_jacobian (bool): Use analytical Jacobian if available (default: True)
            **solver_kwargs: Additional arguments including:
                - n_pools (int): Number of CPU cores for multiprocessing (default: None = all cores)
                - Additional scipy.optimize.curve_fit parameters
        """
        super().__init__(model=model, max_iter=max_iter, tol=tol, verbose=verbose)

        self.method = method
        self.multi_threading = multi_threading
        self.use_jacobian = use_jacobian and hasattr(model, "jacobian")
        self.n_pools = solver_kwargs.pop(
            "n_pools", None
        )  # Number of parallel pools for multithreading
        self.solver_kwargs = solver_kwargs

        # Check p0 and bounds against model.param_names
        if p0 is None:
            # TODO: if p0 and bounds are not provided provide defaults from models
            raise NotImplementedError(
                "Default p0 from model is not yet implemented. Provide p0 explicitly."
            )
        if isinstance(p0[self.model.param_names[0]], (int, float, np.ndarray)):
            validation_utils.validate_parameter_names(p0, self.model.param_names)
            self.p0 = p0
        else:
            raise ValueError(
                "p0 must be a dict with parameter names as keys and initial values as values."
            )
        if bounds is None:
            # set default bounds to (-inf, inf) for all parameters if not provided
            raise NotImplementedError(
                "Default bounds from model is not yet implemented. Provide bounds explicitly."
            )
        elif isinstance(bounds[self.model.param_names[0]], tuple):
            validation_utils.validate_parameter_names(bounds, self.model.param_names)
            self.bounds = bounds
        else:
            raise ValueError(
                "bounds must be a dict with parameter names as keys and (lower, upper) tuples as values."
            )

    def fit(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | dict[str, float] | None = None,
        bounds: (
            tuple[np.ndarray, np.ndarray] | dict[str, tuple[float, float]] | None
        ) = None,
        pixel_fixed_params: dict[str, np.ndarray] | None = None,
        **fit_kwargs,
    ) -> "CurveFitSolver":
        """Fit the model to data using scipy's curve_fit.

        Args:
            xdata (np.ndarray): 1D array of independent variable (e.g., b-values).
            ydata (np.ndarray): 1D or 2D array of observed signals (shape: [n_voxels, n_xdata]).
            p0 (np.ndarray | dict[str, float], optional): Initial parameter guesses (overrides solver defaults if provided).
            bounds (tuple[np.ndarray, np.ndarray] | dict[str, tuple[float, float]], optional): Parameter bounds as (lower, upper) (overrides solver defaults if provided).
            pixel_fixed_params (dict[str, np.ndarray] | None): Per-pixel fixed
                parameter maps.  Each value must be a 1-D array of length
                ``n_pixels``.  When provided, a per-pixel closure is built
                so that each pixel sees its own fixed values.
            **fit_kwargs: Additional keyword arguments for curve_fit (e.g., sigma, absolute_sigma).

        Returns:
            "CurveFitSolver": The fitted solver instance.
        """
        self._reset_state()  # Clear previous fit results

        validation_utils.validate_data_shapes(xdata, ydata)
        n_pixels = ydata.shape[0] if ydata.ndim > 1 else 1

        # Handle multi-voxel fitting
        if ydata.ndim == 1:
            ydata = ydata[np.newaxis, :]  # Convert to 2D for consistent processing

        # prepare p0 and bounds for curve_fit
        p0, bounds = self._validate_p0_and_bounds(p0, bounds, n_pixels)
        logger.info(f"Prepared p0 and bounds for curve_fit for {n_pixels} voxels")

        if self.multi_threading and n_pixels > 1:
            logger.info(f"Starting multi-threaded fitting with {self.n_pools} pools")

        popt, pcov = self._fit_data(
            xdata,
            ydata,
            p0,
            bounds,
            n_pixels,
            pixel_fixed_params=pixel_fixed_params,
        )

        # Store results in self.params_ and self.diagnostics_
        # Determine effective free param names: model.param_names already
        # excludes model-level scalar fixed params; per-pixel fixed params
        # may additionally reduce the free set at fit time.
        free_names = self.model.param_names
        if pixel_fixed_params is not None:
            free_names = [n for n in free_names if n not in pixel_fixed_params]
        self.params_ = {
            name: float(popt[i, 0]) if n_pixels == 1 else popt[i]
            for i, name in enumerate(free_names)
        }
        self.diagnostics_ = {
            "pcov": pcov[0] if n_pixels == 1 else pcov,
            "n_pixels": n_pixels,
        }
        logger.info(f"Fitting complete for {n_pixels} voxel(s).")
        return self

    def _fixed_dict_for_pixel(
        self,
        pixel_idx: int,
        pixel_fixed_params: dict[str, np.ndarray] | None,
    ) -> dict[str, float] | None:
        """Extract a scalar fixed-param dict for one pixel."""
        if pixel_fixed_params is None:
            return None
        return {k: float(v[pixel_idx]) for k, v in pixel_fixed_params.items()}

    def _fit_data(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        n_pixels: int,
        pixel_fixed_params: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit data for all pixels, optionally using multi-threading.

        Args:
            xdata (np.ndarray): 1D array of independent variable.
            ydata (np.ndarray): 2D array of observed signals (shape: [n_pixels, n_xdata]).
            p0 (np.ndarray): Initial parameter guesses (shape: [n_params, n_pixels]).
            bounds (tuple[np.ndarray, np.ndarray]): Parameter bounds as (lower, upper) arrays of shape (n_params, n_pixels).
            n_pixels (int): Number of pixels/voxels to fit.
            pixel_fixed_params (dict[str, np.ndarray] | None): Per-pixel fixed
                parameter maps. Each value must be shape ``(n_pixels,)``.
        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of (popt, pcov) where popt is the optimal parameters array of shape (n_params, n_pixels) and pcov is the covariance array of shape (n_pixels, n_params, n_params).
        """
        if self.multi_threading and n_pixels > 1:
            # Determine number of threads to use
            n_jobs = self.n_pools if self.n_pools > 0 else -1
            logger.info(
                f"Using {n_jobs if n_jobs > 0 else 'all'} CPU cores for parallel fitting"
            )

            # Run parallel fits with progress bar
            results = Parallel(n_jobs=n_jobs, verbose=10 if self.verbose else 0)(
                delayed(self._fit_single_pixel)(
                    xdata,
                    ydata[i],
                    p0[:, i],
                    (bounds[0][:, i], bounds[1][:, i]),
                    pixel_idx=i,
                    pixel_fixed=self._fixed_dict_for_pixel(i, pixel_fixed_params),
                )
                for i in range(n_pixels)
            )
            # Unpack results, replacing None with NaN values
            popt_list = [
                r[0] if r is not None else np.full(p0.shape[0], np.nan) for r in results
            ]
            pcov_list = [
                r[1] if r is not None else np.full((p0.shape[0], p0.shape[0]), np.nan)
                for r in results
            ]
            popt = np.array(popt_list).T  # shape (n_params, n_pixels)
            pcov = np.array(pcov_list)  # shape (n_pixels, n_params, n_params)

        else:
            iterator = tqdm(
                range(n_pixels), desc="Fitting pixels", disable=not self.verbose
            )
            popt_list = []
            pcov_list = []
            for i in iterator:
                popt, pcov = self._fit_single_pixel(
                    xdata,
                    ydata[i],
                    p0[:, i],
                    (bounds[0][:, i], bounds[1][:, i]),
                    pixel_idx=i,
                    pixel_fixed=self._fixed_dict_for_pixel(i, pixel_fixed_params),
                )
                popt_list.append(popt)
                pcov_list.append(pcov)
            popt = np.array(popt_list).T  # shape (n_params, n_pixels)
            pcov = np.array(pcov_list)  # shape (n_pixels, n_params, n_params)
        return popt, pcov

    def _fit_single_pixel(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        pixel_idx: int | None = None,
        pixel_fixed: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a single pixel/voxel using curve_fit.

        Args:
            xdata (np.ndarray): 1D array of independent variable.
            ydata (np.ndarray): 1D array of observed signals for the pixel.
            p0 (np.ndarray): Initial parameter guesses for the pixel.
            bounds (tuple[np.ndarray, np.ndarray]): Parameter bounds as (lower, upper) arrays for the pixel.
            pixel_idx (int | None): Optional index of the pixel being fitted (for logging purposes).
            pixel_fixed (dict[str, float] | None): Per-pixel fixed parameter
                values.  When provided a closure wrapping
                :meth:`model.forward_with_fixed` is passed to ``curve_fit``
                instead of ``model.forward``.
        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of (popt, pcov) where popt is the optimal parameters array for the pixel and pcov is the covariance array for the pixel.
        """
        # Build forward / jacobian callables — use a closure whenever any
        # fixed parameters are in play (per-pixel or model-level scalar).
        fixed = pixel_fixed if pixel_fixed else (self.model.fixed_params or None)
        if fixed:
            fwd = lambda xdata, *p, _f=fixed: self.model.forward_with_fixed(
                xdata, _f, *p
            )
            jac_fn = lambda xdata, *p, _f=fixed: self.model.jacobian_with_fixed(
                xdata, _f, *p
            )
            # When per-pixel fixed params contain keys not in model.fixed_params,
            # p0 and bounds were sized to model.param_names (which includes them).
            # Slice down to the free indices expected by the closure.
            free_idx = self.model._free_indices(fixed)
            if len(free_idx) < len(p0):
                p0 = p0[free_idx]
                bounds = (bounds[0][free_idx], bounds[1][free_idx])
        else:
            fwd = self.model.forward
            jac_fn = None

        jacobian = jac_fn
        try:
            popt, pcov = curve_fit(
                f=fwd,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                bounds=(bounds[0], bounds[1]),
                jac=jacobian,
                method=self.method,
                maxfev=self.max_iter,
                ftol=self.tol,
                **self.solver_kwargs,
            )
            return popt, pcov
        except Exception as e:
            logger.warning(
                f"Fit failed for pixel {pixel_idx if pixel_idx is not None else 'unknown'}: {e}"
            )
            return p0, np.full((len(p0), len(p0)), np.nan)

    def _validate_p0_and_bounds(
        self, p0, bounds, n_pixels: int
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Validate and transform p0 and bounds into the correct format for curve_fit."""
        if p0 is not None:
            if isinstance(p0, dict):
                if isinstance(p0[self.model.param_names[0]], (np.ndarray)):
                    raise ValueError(
                        "p0 should either be a basic dict with scalar values or a single np.ndarray of initial values for all parameters. Spatial non-uniform p0 should be handled separately before calling fit()."
                    )
                elif isinstance(p0[self.model.param_names[0]], (int, float)):
                    p0 = validation_utils.transform_p0(
                        p0, self.model.param_names, n_pixels
                    )
                else:
                    raise ValueError(
                        "p0 dict values must be either all scalars or a single np.ndarray of initial values for all parameters."
                    )
            elif isinstance(p0, np.ndarray):
                pass  # shape validation happens later
            else:
                raise ValueError("p0 must be either a dict or a single np.ndarray.")
        else:
            p0 = validation_utils.transform_p0(
                self.p0, self.model.param_names, n_pixels
            )  # Use solver defaults

        if bounds is not None:
            if isinstance(bounds, dict):
                if isinstance(bounds[self.model.param_names[0]], tuple):
                    bounds = validation_utils.transform_bounds(
                        bounds, self.model.param_names, n_pixels
                    )
                else:
                    raise ValueError(
                        "bounds dict values must be tuples of (lower, upper) for each parameter."
                    )
            elif isinstance(bounds, tuple) and len(bounds) == 2:
                if not all(isinstance(b, np.ndarray) for b in bounds):
                    raise ValueError(
                        "bounds tuple must contain two np.ndarrays (lower, upper) of shape (n_params, n_pixels)."
                    )
                else:
                    pass
            else:
                raise ValueError(
                    "bounds must be either a dict with parameter names as keys and (lower, upper) tuples as values, or a tuple of (lower, upper) np.ndarrays."
                )
        else:
            bounds = validation_utils.transform_bounds(
                self.bounds, self.model.param_names, n_pixels
            )  # Use solver defaults

        # verify that p0 is an np.ndarray of shape (n_params, n_pixels)
        if not isinstance(p0, np.ndarray):
            raise ValueError(
                "p0 must be a single np.ndarray of initial values for all parameters."
            )
        if p0.shape[1] != n_pixels:
            raise ValueError(
                f"p0 shape {p0.shape} does not match number of voxels in ydata {n_pixels}."
            )

        # verify that bounds are tuples of np.ndarrays of shape (n_params, n_pixels)
        if isinstance(bounds, tuple) and len(bounds) == 2:
            if not all(isinstance(b, np.ndarray) for b in bounds):
                raise ValueError(
                    "bounds tuple must contain two np.ndarrays (lower, upper) of shape (n_params, n_pixels)."
                )
            if bounds[0].shape[1] != n_pixels or bounds[1].shape[1] != n_pixels:
                raise ValueError(
                    f"bounds shape {bounds[0].shape} and {bounds[1].shape} do not match number of voxels in ydata {n_pixels}."
                )
        return p0, bounds
