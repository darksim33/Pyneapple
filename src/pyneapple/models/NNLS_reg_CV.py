from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.linalg import norm


def NNLS_reg_fit(basis, H, mu, signal, max_iter):
    """Fitting routine including regularisation option."""

    s, _ = nnls(
        np.matmul(np.concatenate((basis, mu * H)).T, np.concatenate((basis, mu * H))),
        np.matmul(
            np.concatenate((basis, mu * H)).T,
            np.append(signal, np.zeros((len(H[:][1])))),
        ),
        maxiter=max_iter,
    )
    return s


def get_G(basis, H, identity, mu, signal, max_iter):
    """Determining lambda function G."""

    fit = NNLS_reg_fit(basis, H, mu, signal, max_iter)

    # Calculating G with CrossValidation method
    G = (
        norm(signal - np.matmul(basis, fit)) ** 2
        / np.trace(
            identity
            - np.matmul(
                np.matmul(
                    basis,
                    np.linalg.inv(np.matmul(basis.T, basis) + np.matmul(mu * H.T, H)),
                ),
                basis.T,
            )
        )
        ** 2
    )
    return G


def NNLS_reg_CV(
    basis: np.ndarray,
    signal: np.ndarray,
    tol: float,
    max_iter: int,
):
    """
    Regularised NNLS fitting with Cross validation to determine regularisation term.

    Based on CVNNLS.m of the AnalyzeNNLS by Bjarnason et al.

    Parameters:
    ----------
    basis:
        Basis consisting of d_values
    signal:
        signal decay

    Attributes:
    ----------
    mu:
        same as our mu? (old: lambda)
    H:
        reg matrix
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
    G_left = get_G(basis, H, identity, Lambda_left, signal, max_iter)
    G_leftDiff = get_G(basis, H, identity, Lambda_left + tol, signal, max_iter)
    f_left = (G_leftDiff - G_left) / tol

    count = 0
    while abs(Lambda_right - Lambda_left) > tol:
        midpoint = (Lambda_right + Lambda_left) / 2
        # Function (+ delta) and derivative f at middle point
        G_middle = get_G(basis, H, identity, midpoint, signal, max_iter)
        G_middleDiff = get_G(basis, H, identity, midpoint + tol, signal, max_iter)
        f_middle = (G_middleDiff - G_middle) / tol

        if count > 1000:
            print("Original choice of mu might not bracket minimum.")
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
