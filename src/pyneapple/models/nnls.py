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
from loguru import logger

from scipy.optimize import nnls
from .NNLS_reg_CV import NNLS_reg_CV


class NNLS(object):
    """Class to perform NNLS fitting.

    Methods:
        fit: Standard fit for plain and regularized NNLS fitting
        model: Model to create fitted diffusion decay
    """

    @staticmethod
    def fit(
            idx: int | tuple,
            signal: np.ndarray,
            basis: np.ndarray,
            max_iter: int | None,
    ) -> tuple:
        """Standard fit for plain and regularized NNLS fitting.

        Args:
            idx (int): Index of the voxel to be fitted
            signal (np.ndarray): Signal decay to be fitted
            basis (np.ndarray): Basis consisting of d_values
            max_iter (int): Maximum number of iterations
        Returns:
            tuple: Index of the voxel and the fitted spectrum
        """
        try:
            fit, _ = nnls(basis, signal, maxiter=max_iter)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"NNLS fitting failed for index {idx}: {str(e)}")
            fit = np.zeros(basis.shape[1])
        return idx, fit

    @staticmethod
    def model(b_values: np.ndarray, spectrum: np.ndarray, bins: np.ndarray):
        """Model to create fitted diffusion decay.

        Args:
            b_values (np.ndarray): B-values
            spectrum (np.ndarray): Spectrum to be fitted
            bins (np.ndarray): Bins of the spectrum
        """
        signal = 0
        for comp, d in enumerate(bins):
            signal += spectrum[comp] * np.exp(b_values * -d)
        return signal


class NNLSCV(object):
    @staticmethod
    def fit(
            idx: int,
            signal: np.ndarray,
            basis: np.ndarray,
            tol: float | None,
            max_iter: int | None,
    ) -> tuple:
        """Advanced NNLS fit including CV regularisation.

        Args:
            idx (int): Index of the voxel to be fitted
            signal (np.ndarray): Signal decay to be fitted
            basis (np.ndarray): Basis consisting of d_values
            tol (float): Tolerance for the fit
            max_iter (int): Maximum number of iterations
        Returns:
            tuple: Index of the voxel and the fitted spectrum
        """
        try:
            fit, _, _ = NNLS_reg_CV(basis, signal, tol, max_iter)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"NNLS-CV fitting failed for index {idx}: {str(e)}")
            fit = np.zeros(basis.shape[1])
        return idx, fit
