"""Constrained curve-fit solver using scipy.optimize.minimize with SLSQP.

Enforces ``1 - sum(f_i) >= 0`` via an inequality constraint so that
volume fractions in multi-compartment models cannot exceed unity.
Unlike ``CurveFitSolver``, which wraps ``curve_fit`` (and therefore
``least_squares``), this solver uses ``scipy.optimize.minimize`` with the
``SLSQP`` method, which natively supports both box bounds and general
inequality constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from .curvefit import CurveFitSolver


class ConstrainedCurveFitSolver(CurveFitSolver):
    """Solver with inequality-constrained volume fractions.

    Inherits ``CurveFitSolver`` for initialization, p0/bounds handling, and
    multi-voxel orchestration, but replaces the per-pixel fitting routine
    with ``scipy.optimize.minimize(method='SLSQP')`` so that the constraint
    ``1 - sum(f_i) >= 0`` can be imposed.

    Args:
        model: Model instance with ``forward()`` and optionally ``jacobian()``.
        max_iter (int): Maximum optimizer iterations (default 250).
        tol (float): Tolerance for convergence (default 1e-8).
        p0 (dict[str, float] | None): Initial parameter guesses.
        bounds (dict[str, tuple[float, float]] | None): Parameter bounds.
        fraction_constraint (bool): When True, enforce ``sum(f_i) <= 1``
            via an SLSQP inequality constraint. Auto-detects fraction
            parameters whose names start with ``'f'``.
        verbose (bool): Enable verbose output.
        method (str): Ignored (always ``'SLSQP'``); kept for API parity.
        multi_threading (bool): Enable parallel multi-voxel fitting.
        use_jacobian (bool): Use analytical Jacobian if model provides one.
        **solver_kwargs: Extra keyword arguments forwarded to
            ``scipy.optimize.minimize`` (e.g. ``n_pools``).

    Raises:
        ValueError: If ``fraction_constraint=True`` and the model uses
            ``fit_reduced=True``, or has fewer than 2 fraction parameters.
    """

    def __init__(
        self,
        model: Any,
        p0: dict[str, float],
        bounds: dict[str, tuple[float, float]],
        max_iter: int = 250,
        tol: float = 1e-8,
        fraction_constraint: bool = True,
        verbose: bool = False,
        method: str = "SLSQP",
        multi_threading: bool = False,
        use_jacobian: bool = True,
        **solver_kwargs,
    ):
        """Initialize the constrained solver.

        Validates that fraction_constraint is compatible with the model
        configuration, then delegates to ``CurveFitSolver.__init__``.
        """
        # Validate constraint compatibility before parent init
        if fraction_constraint:
            if getattr(model, "fit_reduced", False):
                raise ValueError(
                    "fraction_constraint=True is incompatible with "
                    "fit_reduced=True. Use fraction_constraint=True with "
                    "full fraction mode instead."
                )

        # Parent init handles p0/bounds validation, stores model, etc.
        super().__init__(
            model=model,
            max_iter=max_iter,
            tol=tol,
            p0=p0,
            bounds=bounds,
            verbose=verbose,
            method=method,
            multi_threading=multi_threading,
            use_jacobian=use_jacobian,
            **solver_kwargs,
        )

        # Always override method to SLSQP — this solver requires it.
        self.method = "SLSQP"
        self.fraction_constraint = fraction_constraint

        if fraction_constraint:
            # Auto-detect fraction parameter indices (names starting with 'f',
            # excluding 'S0' which may also be present).
            self._fraction_names = [
                name for name in self.model.param_names if name.startswith("f")
            ]
            if len(self._fraction_names) < 2:
                raise ValueError(
                    "fraction_constraint=True requires at least 2 fraction "
                    f"parameters. Found: {self._fraction_names}"
                )
            self._fraction_indices = [
                self.model.param_names.index(name) for name in self._fraction_names
            ]
            logger.info(
                f"Fraction constraint active on parameters: "
                f"{self._fraction_names} (indices {self._fraction_indices})"
            )
        else:
            self._fraction_names = []
            self._fraction_indices = []

    # ------------------------------------------------------------------
    # Override: single-pixel fitting
    # ------------------------------------------------------------------

    def _fit_single_pixel(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        pixel_idx: int | None = None,
        pixel_fixed: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a single voxel using ``scipy.optimize.minimize`` with SLSQP.

        Minimizes ``0.5 * ||ydata - model.forward(xdata, *p)||^2`` subject
        to box bounds and an optional ``1 - sum(f_i) >= 0`` constraint.

        Args:
            xdata: 1D array of independent variable (e.g., b-values).
            ydata: 1D observed signal for one pixel.
            p0: Initial parameter guesses for this pixel.
            bounds: ``(lower, upper)`` arrays for this pixel.
            pixel_idx: Optional pixel index for logging.
            pixel_fixed: Per-pixel fixed parameter values.

        Returns:
            Tuple ``(popt, pcov)`` where ``popt`` is the optimized parameter
            array and ``pcov`` is the estimated covariance matrix (or NaN
            if the fit fails or Jacobian is unavailable).
        """
        # Build forward / jacobian callables (same logic as parent)
        fixed = pixel_fixed if pixel_fixed else (self.model.fixed_params or None)
        if fixed:
            fwd = lambda xdata, *p, _f=fixed: self.model.forward_with_fixed(
                xdata, _f, *p
            )
            jac_fn = lambda xdata, *p, _f=fixed: self.model.jacobian_with_fixed(
                xdata, _f, *p
            )
            free_idx = self.model._free_indices(fixed)
            if len(free_idx) < len(p0):
                p0 = p0[free_idx]
                bounds = (bounds[0][free_idx], bounds[1][free_idx])
        else:
            fwd = self.model.forward
            jac_fn = None

        # Determine fraction indices for this pixel — when per-pixel fixed
        # params remove some fraction params, recompute from the free set.
        if self.fraction_constraint:
            if fixed:
                free_names = [n for n in self.model._all_param_names if n not in fixed]
            else:
                free_names = list(self.model.param_names)
            frac_idx = [i for i, n in enumerate(free_names) if n.startswith("f")]
        else:
            frac_idx = []

        # ----- objective: 0.5 * sum(residual^2) -----
        def objective(p):
            residual = ydata - fwd(xdata, *p)
            return 0.5 * np.dot(residual, residual)

        # ----- gradient (if analytical Jacobian available) -----
        use_jac = self.use_jacobian and (
            jac_fn is not None or hasattr(self.model, "jacobian")
        )

        def gradient(p):
            residual = ydata - fwd(xdata, *p)
            if jac_fn is not None:
                J = jac_fn(xdata, *p)
            else:
                J = self.model.jacobian(xdata, *p)
            if J is None:
                # Fallback: let SLSQP use finite differences
                return None
            # grad = -J^T @ r
            return -J.T @ residual

        jac_arg = gradient if use_jac else None

        # ----- bounds in scipy.optimize.minimize format -----
        scipy_bounds = list(zip(bounds[0], bounds[1]))

        # ----- constraints -----
        constraints = []
        if self.fraction_constraint and len(frac_idx) >= 2:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda p, _idx=frac_idx: 1.0 - sum(p[i] for i in _idx),
                }
            )

        try:
            result = minimize(
                fun=objective,
                x0=p0,
                jac=jac_arg,
                method="SLSQP",
                bounds=scipy_bounds,
                constraints=constraints,
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.tol,
                    "disp": self.verbose,
                },
            )
            popt = result.x

            # Estimate covariance from Jacobian at solution
            pcov = self._estimate_covariance(xdata, ydata, popt, fwd, jac_fn)

            if not result.success:
                logger.warning(
                    f"Pixel {pixel_idx}: optimizer did not converge — {result.message}"
                )

            return popt, pcov

        except Exception as e:
            logger.warning(
                f"Fit failed for pixel "
                f"{pixel_idx if pixel_idx is not None else 'unknown'}: {e}"
            )
            return p0, np.full((len(p0), len(p0)), np.nan)

    def _estimate_covariance(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        popt: np.ndarray,
        fwd,
        jac_fn,
    ) -> np.ndarray:
        """Estimate parameter covariance from the Jacobian at the solution.

        Uses the Gauss-Newton approximation:
        ``pcov = s^2 * inv(J^T @ J)`` where ``s^2`` is the residual
        variance estimate.

        Args:
            xdata: Independent variable.
            ydata: Observed data.
            popt: Optimized parameters.
            fwd: Forward model callable.
            jac_fn: Jacobian callable (or None).

        Returns:
            Covariance matrix of shape ``(n_params, n_params)``, or NaN
            array if computation fails.
        """
        n_params = len(popt)
        try:
            if jac_fn is not None:
                J = jac_fn(xdata, *popt)
            elif hasattr(self.model, "jacobian"):
                J = self.model.jacobian(xdata, *popt)
            else:
                J = None

            if J is None:
                return np.full((n_params, n_params), np.nan)

            residual = ydata - fwd(xdata, *popt)
            n_data = len(ydata)
            dof = max(n_data - n_params, 1)
            s_sq = np.dot(residual, residual) / dof
            JtJ = J.T @ J
            pcov = s_sq * np.linalg.inv(JtJ)
            return pcov
        except (np.linalg.LinAlgError, ValueError):
            return np.full((n_params, n_params), np.nan)
