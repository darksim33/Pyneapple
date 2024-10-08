from __future__ import annotations

import numpy as np
import time

from scipy.optimize import nnls
from .NNLS_reg_CV import NNLS_reg_CV


class NNLS(object):
    @staticmethod
    def fit(
        idx: int | tuple,
        signal: np.ndarray,
        basis: np.ndarray,
        max_iter: int | None,
    ) -> tuple:
        """Standard fit for plain and regularised NNLS fitting."""
        try:
            fit, _ = nnls(basis, signal, maxiter=max_iter)
        except (RuntimeError, ValueError):
            fit = np.zeros(basis.shape[1])
        return idx, fit

    @staticmethod
    def model(b_values: np.ndarray, spectrum: np.ndarray, bins: np.ndarray):
        """Model to create fitted diffusion decay."""
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
        """Advanced NNLS fit including CV regularisation."""
        try:
            fit, _, _ = NNLS_reg_CV(basis, signal, tol, max_iter)
        except (RuntimeError, ValueError):
            fit = np.zeros(basis.shape[1])
        return idx, fit
