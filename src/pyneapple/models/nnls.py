""" Module to perform NNLS fitting.

Classes:
    NNLS: Class to perform NNLS fitting
    NNLSCV: Class to perform NNLS fitting with CV regularisation
Methods:
    fit: Standard fit for plain and regularized NNLS fitting
    model: Model to create fitted diffusion decay
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ..utils.logger import logger
from .model import AbstractFitModel
from .NNLS_reg_CV import NNLS_reg_CV


class NNLSModel(AbstractFitModel):
    """Class to perform NNLS fitting.

    Methods:
        fit: Standard fit for plain and regularized NNLS fitting
        model: Model to create fitted diffusion decay
    """

    def __init__(self, **kwargs):
        """Initialize the NNLS model."""
        super().__init__(name="NNLS", **kwargs)
        self._args = None

    @property
    def args(self):
        """Get the arguments used in the current configured model."""
        return self._args

    def model(self, b_values: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Model to create fitted diffusion decay.

        Since the NNLS returns a spectrum, this function calculates the decay signal resulting from the spectrum and the
        b-values. Unlike the IVIM models this model is not used for fitting.

        Args:
            b_values (np.ndarray): B-values
            **kwargs:
                spectrum (np.ndarray): Spectrum to be fitted (!not optional!)
                bins (np.ndarray): Bins of the spectrum (!not optional!)
                """
        signal = 0
        spectrum = kwargs.get("spectrum", None)
        if spectrum is None:
            error_msg = "Spectrum must be provided in kwargs as 'spectrum'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        bins = kwargs.get("bins", None)
        if bins is None:
            error_msg = "Bins must be provided in kwargs as 'bins'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        for comp, d in enumerate(bins):
            signal += spectrum[comp] * np.exp(b_values * -d)
        return signal

    @staticmethod
    def _get_fit_args(**kwargs) -> tuple[np.ndarray, int]:
        """Get the fitting arguments from kwargs.

        Args:
            **kwargs:
                basis (np.ndarray): Basis consisting of d_values
                max_iter (int): Maximum number of iterations
        Returns:
            tuple: Basis and maximum number of iterations
        """
        basis = kwargs.get("basis", None)
        if basis is None:
            error_msg = "Basis must be provided in kwargs as 'basis'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        max_iter = kwargs.get("max_iter", None)
        if max_iter is None:
            error_msg = "max_iter must be provided in kwargs as 'max_iter'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        return basis, max_iter

    def fit(
            self,
            idx: int | tuple,
            signal: np.ndarray,
            *args,
            **kwargs,
    ) -> tuple:
        """Standard fit for plain and regularized NNLS fitting.

        Args:
            idx (int): Index of the voxel to be fitted
            signal (np.ndarray): Signal decay to be fitted
            **kwargs:
                basis (np.ndarray): Basis consisting of d_values
                max_iter (int): Maximum number of iterations
        Returns:
            tuple: Index of the voxel and the fitted spectrum
        """
        basis, max_iter = self._get_fit_args(**kwargs)

        try:
            fit, _ = nnls(basis, signal, maxiter=max_iter)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"NNLS fitting failed for index {idx}: {str(e)}")
            fit = np.zeros(basis.shape[1])
        return idx, fit


class NNLSCVModel(NNLSModel):
    def __init__(self, **kwargs):
        """Initialize the NNLSCV model."""
        super().__init__(name="NNLSCV", **kwargs)
        self._args = None

    def fit(
            self,
            idx: int,
            signal: np.ndarray,
            *args, **kwargs
    ) -> tuple:
        """Advanced NNLS fit including CV regularisation.

        Args:
            idx (int): Index of the voxel to be fitted
            signal (np.ndarray): Signal decay to be fitted
            **kwargs:
                basis (np.ndarray): Basis consisting of d_values
                tol (float): Tolerance for the fit
                max_iter (int): Maximum number of iterations
        Returns:
            tuple: Index of the voxel and the fitted spectrum
        """
        basis, max_iter = self._get_fit_args(**kwargs)
        tol = kwargs.get("tol", None)
        if tol is None:
            error_msg = "tol must be provided in kwargs as 'tol'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            fit, _, _ = NNLS_reg_CV(basis, signal, tol, max_iter)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"NNLS-CV fitting failed for index {idx}: {str(e)}")
            fit = np.zeros(basis.shape[1])
        return idx, fit
