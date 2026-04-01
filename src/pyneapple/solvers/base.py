"""Base solver interface for optimization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any

from loguru import logger


class BaseSolver(ABC):
    """Abstract base class for optimization solvers."""

    def __init__(
        self,
        model: Any,
        max_iter: int = 250,
        tol: float = 1e-8,
        verbose: bool = False,
        **solver_kwargs,
    ):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.diagnostics_: dict[str, Any] = {}
        self.params_: dict[str, Any] = {}

        if self.verbose:
            logger.info(
                f"Initialized {self.__class__.__name__} with solver_kwargs={solver_kwargs}"
            )

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseSolver":
        """Fit the optimization model."""
        return self

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics information about the solver."""
        if len(self.diagnostics_) == 0:
            error_msg = "No diagnostics available. Ensure fit() has been called and diagnostics are stored."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return self.diagnostics_.copy()

    def get_params(self) -> dict[str, Any]:
        """Return the fitted parameters."""
        if len(self.params_) == 0:
            error_msg = "No parameters available. Ensure fit() has been called and parameters are stored."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return self.params_.copy()

    def _reset_state(self):
        """Reset solver state before a new fit."""
        self.diagnostics_ = {}
        self.params_ = {}
