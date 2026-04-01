"""Pure functions for NNLS basis matrix and regularization construction.

Stateless functions for building the components needed by NNLS solvers:
- Diffusion coefficient bins (log-spaced)
- Basis matrix construction
- Regularization matrices (orders 0-3)
- Signal reconstruction from spectrum

These functions are used by both NNLSSolver/NNLSCVSolver classes.
"""

from __future__ import annotations

import numpy as np


def get_bins(d_min: float, d_max: float, n_bins: int) -> np.ndarray:
    """Create logarithmically spaced diffusion coefficient bins.

    Args:
        d_min: Minimum diffusion coefficient
        d_max: Maximum diffusion coefficient
        n_bins: Number of bins

    Returns:
        np.ndarray: Log-spaced D values, shape (n_bins,)
    """
    return np.logspace(np.log10(d_min), np.log10(d_max), n_bins)


def get_basis(bvalues: np.ndarray, d_values: np.ndarray) -> np.ndarray:
    r"""Construct the NNLS basis matrix: :math:`B_{ij} = e^{-b_i \cdot D_j}`.

    Args:
        bvalues: B-values, shape (n_measurements,)
        d_values: Diffusion coefficient bins, shape (n_bins,)

    Returns:
        np.ndarray: Basis matrix, shape (n_measurements, n_bins)
    """
    b = bvalues.reshape(-1, 1)  # (n_measurements, 1)
    d = d_values.reshape(1, -1)  # (1, n_bins)
    return np.exp(-b * d)


def regularization_matrix(n_bins: int, order: int, mu: float = 1.0) -> np.ndarray:
    """Construct a Tikhonov regularization matrix.

    Args:
        n_bins: Number of bins (matrix will be n_bins x n_bins)
        order: Regularization order:
            - 0: No regularization (zero matrix)
            - 1: First-order difference (predecessor weighting)
            - 2: Second-order difference (nearest neighbor weighting)
            - 3: Extended second-order (first + second nearest neighbor)
        mu: Regularization strength parameter

    Returns:
        np.ndarray: Regularization matrix, shape (n_bins, n_bins)

    Raises:
        NotImplementedError: If order > 3
    """
    if order == 0:
        return np.zeros((n_bins, n_bins))
    elif order == 1:
        return (np.diag(np.full(n_bins, -1.0)) + np.diag(np.ones(n_bins - 1), 1)) * mu
    elif order == 2:
        return (
            np.diag(np.ones(n_bins - 1), -1)
            + np.diag(np.full(n_bins, -2.0))
            + np.diag(np.ones(n_bins - 1), 1)
        ) * mu
    elif order == 3:
        return (
            np.diag(np.ones(n_bins - 2), -2)
            + np.diag(np.full(n_bins - 1, 2.0), -1)
            + np.diag(np.full(n_bins, -6.0))
            + np.diag(np.full(n_bins - 1, 2.0), 1)
            + np.diag(np.ones(n_bins - 2), 2)
        ) * mu
    else:
        raise NotImplementedError(
            f"Regularization order {order} not supported. Use 0-3."
        )


def curvature_matrix(n_bins: int) -> np.ndarray:
    """Construct the curvature matrix H for cross-validation NNLS.

    Used by NNLSCVSolver for bisection search:
        H = -2I + diag(1, k=1) + diag(1, k=-1)

    Args:
        n_bins: Number of bins

    Returns:
        np.ndarray: Curvature matrix, shape (n_bins, n_bins)
    """
    return np.array(
        -2 * np.identity(n_bins)
        + np.diag(np.ones(n_bins - 1), 1)
        + np.diag(np.ones(n_bins - 1), -1)
    )


def reconstruct_signal(
    bvalues: np.ndarray,
    spectrum: np.ndarray,
    d_values: np.ndarray,
) -> np.ndarray:
    r"""Reconstruct signal from fitted spectrum: :math:`S = B \cdot s`.

    Where B is the basis matrix and s is the spectrum.

    Args:
        bvalues: B-values, shape (n_measurements,)
        spectrum: Fitted spectrum, shape (n_bins,)
        d_values: Diffusion coefficient bins, shape (n_bins,)

    Returns:
        np.ndarray: Reconstructed signal, shape (n_measurements,)
    """
    basis = get_basis(bvalues, d_values)
    return basis @ spectrum


def build_regularized_basis(
    basis: np.ndarray, n_bins: int, order: int, mu: float
) -> np.ndarray:
    """Build a basis matrix with regularization appended.

    Concatenates the basis matrix with the regularization matrix:
        [basis; mu * R]

    Args:
        basis: Plain basis matrix, shape (n_measurements, n_bins)
        n_bins: Number of bins
        order: Regularization order (0-3)
        mu: Regularization strength

    Returns:
        np.ndarray: Regularized basis, shape (n_measurements + n_bins, n_bins)
    """
    reg = regularization_matrix(n_bins, order, mu)
    return np.concatenate((basis, reg), axis=0)
