import numpy as np
from scipy.optimize import nnls
from scipy.linalg import norm


def NNLS_fit(A, H, Lambda, signal):
    # Regularised fitting routine

    s, _ = nnls(
        np.matmul(np.concatenate((A, Lambda * H)).T, np.concatenate((A, Lambda * H))),
        np.matmul(
            np.concatenate((A, Lambda * H)).T,
            np.append(signal, np.zeros((len(H[:][1])))),
        ),
        maxiter=2000,  # 200
    )
    return s


def get_G(A, H, In, Lambda, signal):
    # Determining lambda function G

    fit = NNLS_fit(A, H, Lambda, signal)
    # Calculating G with CrossValidation method
    G = (
        norm(signal - np.matmul(A, fit)) ** 2
        / np.trace(
            In
            - np.matmul(
                np.matmul(
                    A, np.linalg.inv(np.matmul(A.T, A) + np.matmul(Lambda * H.T, H))
                ),
                A.T,
            )
        )
        ** 2
    )
    return G


def NNLS_reg_CV(basis: np.ndarray, signal: np.ndarray, tol: float | None = 0.0001):
    # Regularised NNLS fitting based on CVNNLS.m of the AnalyzeNNLS by Bjarnason et al.
    # With Cross validation to determine regularisation term

    # Identity matrix
    In = np.identity(len(signal))

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
    G_left = get_G(basis, H, In, Lambda_left, signal)
    G_leftDiff = get_G(basis, H, In, Lambda_left + tol, signal)
    f_left = (G_leftDiff - G_left) / tol

    count = 0
    while abs(Lambda_right - Lambda_left) > tol:
        midpoint = (Lambda_right + Lambda_left) / 2
        # Function (+ delta) and derivative f at middle point
        G_middle = get_G(basis, H, In, midpoint, signal)
        G_middleDiff = get_G(basis, H, In, midpoint + tol, signal)
        f_middle = (G_middleDiff - G_middle) / tol

        if count > 100:
            print("Original choice of Lambda might not bracket minimum.")
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
    Lambda = midpoint
    results = NNLS_fit(basis, H, Lambda, signal)

    # Determine chi2_min
    [_, resnorm_min] = nnls(basis, signal)

    # Determine chi2_smooth
    y_recon = np.matmul(basis, results)
    resid = signal - y_recon
    resnorm_smooth = np.sum(np.multiply(resid, resid))
    chi = resnorm_smooth / resnorm_min

    return results, chi, resid
