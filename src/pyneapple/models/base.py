"""Base model classes following scikit-learn estimator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from loguru import logger

from ..utility.validation import validate_fixed_params


class BaseModel(ABC):
    """Abstract base class for all diffusion MRI models.

    This class defines the minimal interface that all models must implement.
    Models represent the forward physics (signal equations) and can be fitted
    using different solver backends.

    Two main model categories:
    - ParametricModel: Discrete parameters (e.g., monoexponential, IVIM)
    - DistributionModel: Continuous distributions (e.g., NNLS)
    """

    def __init__(self, **model_kwargs: Any):
        """Initialize the model.

        Args:
            **model_kwargs: Model-specific configuration
        """
        self.model_kwargs = model_kwargs

    @abstractmethod
    def forward(self, xdata: np.ndarray, *params: float) -> np.ndarray:
        """Evaluate the forward model.

        Args:
            xdata: Independent variable (e.g., b-values)
            params: Model parameters

        Returns:
            Model prediction (signal)
        """
        pass

    def jacobian(self, xdata: np.ndarray, *params: float) -> np.ndarray | None:
        """Analytical Jacobian (optional).

        Args:
            xdata: Independent variable
            params: Model parameters

        Returns:
            Partial derivatives, or None if not implemented
        """
        pass

    def residual(
        self,
        xdata: np.ndarray,
        measured_signal: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Calculate residuals: observed - predicted.

        Args:
            xdata: Independent variable
            measured_signal: Observed data
            params: Model parameters

        Returns:
            Residuals array
        """
        return measured_signal - self.forward(xdata, *params)


class ParametricModel(BaseModel):
    """Base class for parametric models with discrete parameters.

    Extends BaseModel with utilities for parameter conversion between
    dict and array formats, required for scipy solvers.

    Models with discrete parameters like:
    - Monoexponential: {'S0', 'D'}
    - Biexponential: {'S0', 'D1', 'f1', 'D2', 'f2'}

    Provides:
    - Fixed parameter support (``fixed_params``)
    - Parameter dict ↔ array conversion
    - Jacobian dict ↔ array conversion
    - Bounds dict ↔ array conversion
    - Wrappers for scipy.optimize.curve_fit
    """

    def __init__(
        self,
        fixed_params: dict[str, float] | None = None,
        **model_kwargs,
    ):
        """Initialize the parametric model.

        Args:
            fixed_params: Parameters to hold constant during fitting, as
                ``{name: value}``.  Keys must be valid parameter names
                returned by ``_all_param_names``.
            **model_kwargs: Model-specific configuration forwarded to
                subclasses and stored on ``self.model_kwargs``.
        """
        super().__init__(**model_kwargs)
        self.fixed_params: dict[str, float] = dict(fixed_params) if fixed_params else {}

    def _validate_fixed_params(self) -> None:
        """Validate ``fixed_params`` against ``_all_param_names``.

        Concrete subclasses should call this at the end of their ``__init__``
        after all mode-dependent attributes are set.
        """
        if self.fixed_params:
            validate_fixed_params(self.fixed_params, self._all_param_names)

    # ------------------------------------------------------------------
    # Abstract interface that concrete subclasses must implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _all_param_names(self) -> list[str]:
        """Return the full ordered list of ALL parameter names.

        Includes parameters that may be fixed.  The positional order here
        defines what ``forward`` and ``jacobian`` receive.
        """
        pass

    # Concrete subclasses must also implement:
    #   forward(xdata, *all_params)  — inherited abstract from BaseModel
    #   jacobian(xdata, *all_params) — optional override from BaseModel

    # ------------------------------------------------------------------
    # Fixed-param helpers (solver-facing)
    # ------------------------------------------------------------------

    def _inject_fixed(
        self, free_params: tuple, fixed: dict[str, float]
    ) -> tuple:
        """Reconstruct the full parameter tuple by merging free and fixed values.

        Args:
            free_params: Values for the free (non-fixed) parameters in
                ``param_names`` order.
            fixed: Dictionary of ``{name: value}`` for fixed parameters.

        Returns:
            Full parameter tuple in ``_all_param_names`` order suitable for
            passing to ``forward`` / ``jacobian``.
        """
        if not fixed:
            return free_params
        free_iter = iter(free_params)
        return tuple(
            float(fixed[name]) if name in fixed else next(free_iter)
            for name in self._all_param_names
        )

    def _free_indices(self, fixed: dict[str, float]) -> list[int]:
        """Return column indices of free (non-fixed) parameters."""
        return [
            i
            for i, name in enumerate(self._all_param_names)
            if name not in fixed
        ]

    @property
    def param_names(self) -> list[str]:
        """Return ordered list of *free* (non-fixed) parameter names.

        Parameters present in ``fixed_params`` are excluded.  This is the
        list the solver uses to size its arrays and label results.
        """
        if not self.fixed_params:
            return self._all_param_names
        return [p for p in self._all_param_names if p not in self.fixed_params]

    @property
    def n_params(self) -> int:
        """Return number of free parameters."""
        return len(self.param_names)

    def forward_with_fixed(
        self, xdata: np.ndarray, fixed_dict: dict[str, float], *free_params: float
    ) -> np.ndarray:
        """Forward model with injection of fixed parameter values.

        Merges *free_params* with *fixed_dict* into a full parameter tuple
        and delegates to :meth:`forward`.

        Args:
            xdata: Independent variable (e.g., b-values).
            fixed_dict: Fixed parameter values ``{name: value}``.
            *free_params: Free parameter values in ``param_names`` order.

        Returns:
            Model prediction (signal).
        """
        all_params = self._inject_fixed(free_params, fixed_dict)
        return self.forward(xdata, *all_params)

    def jacobian_with_fixed(
        self, xdata: np.ndarray, fixed_dict: dict[str, float], *free_params: float
    ) -> np.ndarray | None:
        """Jacobian with injection of fixed parameter values.

        Merges *free_params* with *fixed_dict*, calls :meth:`jacobian`, then
        slices away columns corresponding to fixed parameters.

        Args:
            xdata: Independent variable.
            fixed_dict: Fixed parameter values ``{name: value}``.
            *free_params: Free parameter values in ``param_names`` order.

        Returns:
            ``(n_xdata, n_free_params)`` Jacobian array, or ``None``.
        """
        all_params = self._inject_fixed(free_params, fixed_dict)
        jac_full = self.jacobian(xdata, *all_params)
        if jac_full is None:
            return None
        return jac_full[:, self._free_indices(fixed_dict)]

    def precondition(
        self, jacobian: np.ndarray, method: str = "diagonal"
    ) -> np.ndarray | None:
        """Precondition the Jacobian for better numerical conditioning.

        Balances contributions of parameters with different scales
        (e.g., S0 ~ 1000 vs D ~ 0.001).

        Args:
            jacobian: Jacobian array of shape (n_bvalues, n_params).
            method: Preconditioning method (``'diagonal'`` or ``'none'``).

        Returns:
            Preconditioned Jacobian with same structure as input.
        """
        if method == "none":
            preconditioned = jacobian

        elif method == "diagonal":
            # Precondition the Jacobian matrix directly
            norm = np.sqrt(np.sum(jacobian**2, axis=0, keepdims=True))
            norm = np.where(norm > 1e-12, norm, 1.0)
            preconditioned = jacobian / norm

        else:
            raise ValueError(
                f"Unknown preconditioning method: {method}. "
                f"Supported: 'diagonal', 'none'"
            )

        logger.debug(f"Applied {method} preconditioning to Jacobian.")
        return preconditioned

    def validate_params(self, params: dict[str, float]) -> np.ndarray:
        """Validate parameter dictionary.

        Checks that all required parameter names are present, warns about
        extras, and returns values in ``param_names`` order.

        Args:
            params: Dictionary of ``{name: value}`` for all free parameters.

        Returns:
            np.ndarray: Parameter values in ``param_names`` order.

        Raises:
            ValueError: If any required parameter is missing.
        """
        missing = set(self.param_names) - set(params.keys())
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"Required: {self.param_names}"
            )

        extra = set(params.keys()) - set(self.param_names)
        if extra:
            logger.warning(f"Extra parameters will be ignored: {extra}")

        # get all values in right order
        values = [params[name] for name in self.param_names]
        return np.array(values)

    def validate_bounds(
        self, bounds: dict[str, tuple[float | np.ndarray, float | np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate parameter bounds.

        Checks that all required parameter names are present, bounds are tuples
        of (lower, upper), and shapes are consistent.

        Args:
            bounds: Dictionary of parameter bounds as {param: (lower, upper)}.

        Raises:
            ValueError: If bounds are missing, malformed, or shapes are inconsistent.
        """
        missing = set(self.param_names) - set(bounds.keys())
        if missing:
            raise ValueError(
                f"Missing bounds for required parameters: {missing}. "
                f"Required: {self.param_names}"
            )

        extra = set(bounds.keys()) - set(self.param_names)
        if extra:
            logger.warning(f"Extra bounds will be ignored: {extra}")

        lower = np.array([bounds[name][0] for name in self.param_names])
        upper = np.array([bounds[name][1] for name in self.param_names])
        return (lower, upper)


class DistributionModel(BaseModel):
    """Base class for models representing continuous distributions.

    Examples include NNLS models where the output is a distribution over a
    range of parameters (e.g., diffusion coefficients) rather than discrete
    parameter values.

    Subclasses must implement:
    - ``bins``: the discrete parameter grid.
    - ``get_basis``: the forward dictionary matrix mapping bins to measurements.

    ``forward`` is implemented here as signal reconstruction from a spectrum
    (basis @ spectrum), so subclasses only need to provide ``bins`` and
    ``get_basis``.
    """

    @property
    @abstractmethod
    def bins(self) -> np.ndarray:
        """Return the discrete parameter bins for the distribution.

        Returns:
            Array of shape (n_bins,) with the discrete parameter values.
        """
        pass

    @abstractmethod
    def get_basis(self, xdata: np.ndarray) -> np.ndarray:
        """Construct the basis (dictionary) matrix.

        Args:
            xdata: Independent variable, shape (n_measurements,).

        Returns:
            Basis matrix of shape (n_measurements, n_bins).
        """
        pass

    def forward(self, xdata: np.ndarray, *spectrum: float) -> np.ndarray:
        """Reconstruct signal from a spectrum of bin coefficients.

        Args:
            xdata: Independent variable (e.g., b-values), shape (n_measurements,).
            *spectrum: Coefficients, one per bin.

        Returns:
            Reconstructed signal of shape (n_measurements,).
        """
        return self.get_basis(xdata) @ np.asarray(spectrum)
