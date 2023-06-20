import numpy as np
from scipy.optimize import nnls


def NNLSreg(
    basis: np.ndarray,
    signal: np.ndarray,
    # d_values: np.ndarray,
    reg_order: int,
    mu: float,
    maxiter: int,
):
    """
    NNLS Analysis with regularisation

    basis: np.ndarray, shape = (number datapoints, number bins)

    mu: float, is the regularisation factor

    """

    n_data = basis.shape[0]
    n_bins = basis.shape[1]

    # create new basis and signal
    basis_new = np.zeros([n_data + n_bins, n_bins])
    signal_new = np.zeros([n_data + n_bins])
    # add current set to new one
    basis_new[0:n_data, 0:n_bins] = basis
    signal_new[0:n_data] = signal

    for i in range(n_bins, (n_bins + n_data), 1):
        # idx_data is iterator for the datapoints
        # since the new basis is already filled with the basis set it only needs to iterate beyond that
        for j in range(n_bins):
            # idx_bin is the iterator for the bins
            basis_new[i, j] = 0
            if reg_order == 0:
                # no weighting
                if i - n_data == j:
                    basis_new[i, j] = 1.0 * mu
            elif reg_order == 1:
                # weighting with the predecessor
                if i - n_data == j:
                    basis_new[i, j] = -1.0 * mu
                elif i - n_data == j + 1:
                    basis_new[i, j] = 1.0 * mu
            elif reg_order == 2:
                # weighting of the nearest neighbours
                if i - n_data == j - 1:
                    basis_new[i, j] = 1.0 * mu
                elif i - n_data == j:
                    basis_new[i, j] = -2.0 * mu
                elif i - n_data == j + 1:
                    basis_new[i, j] = 1.0 * mu
            elif reg_order == 3:
                # weighting of the first and second nearest neighbours
                if i - n_data == j - 2:
                    basis_new[i, j] = 1.0 * mu
                elif i - n_data == j - 1:
                    basis_new[i, j] = 2.0 * mu
                elif i - n_data == j:
                    basis_new[i, j] = -6.0 * mu
                elif i - n_data == j + 1:
                    basis_new[i, j] = 2.0 * mu
                elif i - n_data == j + 2:
                    basis_new[i, j] = 1.0 * mu

    fit, _ = nnls(basis_new, signal_new, maxiter=maxiter)
