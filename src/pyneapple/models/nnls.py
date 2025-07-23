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
from scipy.linalg import norm

from ..utils.logger import logger
from .model import AbstractFitModel


class NNLSModel(AbstractFitModel):
    """Class to perform NNLS fitting.

    Methods:
        fit: Standard fit for plain and regularized NNLS fitting
        model: Model to create fitted diffusion decay
    """

    def __init__(self, name: str = "NNLS", **kwargs):
        """Initialize the NNLS model."""
        super().__init__(name=name, **kwargs)
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
    def __init__(self, name: str = "NNLSCV", **kwargs):
        """Initialize the NNLSCV model."""
        super().__init__(name=name, **kwargs)
        self._args = None

    def fit(self, idx: int, signal: np.ndarray, *args, **kwargs) -> tuple:
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
            fit, _, _ = self.nnls_reg_cv(basis, signal, tol, max_iter)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"NNLS-CV fitting failed for index {idx}: {str(e)}")
            fit = np.zeros(basis.shape[1])
        return idx, fit

    def nnls_reg_cv(
        self,
        basis: np.ndarray,
        signal: np.ndarray,
        tol: float,
        max_iter: int,
    ):
        """Regularised NNLS fitting with Cross validation to determine regularisation term.

        Based on CVNNLS.m of the AnalyzeNNLS by Bjarnason et al.

        Args:
            basis (np.ndarray):Basis consisting of d_values
            signal (np.ndarray): Signal decay to be fitted

        Attributes:
            mu: same as our mu? (old: lambda)
            H: reg matrix
        """

        # Identity matrix
        identity = np.identity(len(signal))

        # Curvature
        n_bins = len(basis[1][:])
        H = np.array(
            -2 * np.identity(n_bins)
            + np.diag(np.ones(n_bins - 1), 1)
            + np.diag(np.ones(n_bins - 1), -1)
        )

        Lambda_left = 0.00001
        Lambda_right = 8
        midpoint = (Lambda_right + Lambda_left) / 2

        # Function (+ delta) and derivative f at left point
        G_left = self._get_G(basis, H, identity, Lambda_left, signal, max_iter)
        G_leftDiff = self._get_G(
            basis, H, identity, Lambda_left + tol, signal, max_iter
        )
        f_left = (G_leftDiff - G_left) / tol

        count = 0
        while abs(Lambda_right - Lambda_left) > tol:
            midpoint = (Lambda_right + Lambda_left) / 2
            # Function (+ delta) and derivative f at middle point
            G_middle = self._get_G(basis, H, identity, midpoint, signal, max_iter)
            G_middleDiff = self._get_G(
                basis, H, identity, midpoint + tol, signal, max_iter
            )
            f_middle = (G_middleDiff - G_middle) / tol

            if count > 1000:
                logger.warning("Original choice of mu might not bracket minimum.")
                break

            # Continue with logic
            if f_left * f_middle > 0:
                # Throw away left half
                Lambda_left = midpoint
                f_left = f_middle
            else:
                # Throw away right half
                Lambda_right = midpoint
            count = +1

        # NNLS fit of found minimum
        mu = midpoint
        fit_result = NNLS_reg_fit(basis, H, mu, signal, max_iter)
        # Change fitting to standard NNLSParams.fit function for consistency
        # _, results_test = Model.NNLS.fit(1, signal, basis, 200)

        # Determine chi2_min
        [_, resnorm_min] = nnls(basis, signal)

        # Determine chi2_smooth
        y_recon = np.matmul(basis, fit_result)
        resid = signal - y_recon
        resnorm_smooth = np.sum(np.multiply(resid, resid))
        chi = resnorm_smooth / resnorm_min

        return fit_result, chi, resid

    def _get_G(self, basis, H, identity, mu, signal, max_iter):
        """Determining lambda function G."""

        fit = self.nnls_reg_fit(basis, H, mu, signal, max_iter)

        # Calculating G with CrossValidation method
        G = (
            norm(signal - np.matmul(basis, fit)) ** 2
            / np.trace(
                identity
                - np.matmul(
                    np.matmul(
                        basis,
                        np.linalg.inv(
                            np.matmul(basis.T, basis) + np.matmul(mu * H.T, H)
                        ),
                    ),
                    basis.T,
                )
            )
            ** 2
        )
        return G

    @staticmethod
    def nnls_reg_fit(basis, H, mu, signal, max_iter):
        """Fitting routine including regularisation option."""

        s, _ = nnls(
            np.matmul(
                np.concatenate((basis, mu * H)).T, np.concatenate((basis, mu * H))
            ),
            np.matmul(
                np.concatenate((basis, mu * H)).T,
                np.append(signal, np.zeros((len(H[:][1])))),
            ),
            maxiter=max_iter,
        )
        return s
