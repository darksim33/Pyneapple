import numpy as np
from scipy.optimize import nnls
from scipy.linalg import norm


# renamed file and main function
# added max iter to nnls


def nnls_fit(A, H, Lambda, signal):
    # Regularised fitting routine

    s, _ = nnls(
        np.matmul(np.concatenate((A, Lambda * H)).T, np.concatenate((A, Lambda * H))),
        np.matmul(
            np.concatenate((A, Lambda * H)).T,
            np.append(signal, np.zeros((len(H[:][1])))),
        ),
        maxiter=200,
    )
    return s


def getG(A, H, identity, Lambda, signal):
    # Determining lambda function G

    NNLS_fit = nnls_fit(A, H, Lambda, signal)
    # Calculating G with CrossValidation method
    G = (norm(signal - np.matmul(A, NNLS_fit)) ** 2 /
         np.trace(
             identity - np.matmul(
                 np.matmul(
                     A, np.linalg.inv(np.matmul(A.T, A) + np.matmul(Lambda * H.T, H))),
                 A.T, )) ** 2)
    return G


def NNLSregCV(basis: np.ndarray, signal: np.ndarray, tol: float = 0.0001):
    # Regularised NNLS fitting based on CVNNLS.m of the AnalyzeNNLS by Bjarnason et al.
    # With Cross validation to determine regularisation term

    # Identity matrix
    identity = np.identity(len(signal))

    # Curvature
    n_basis = len(basis[1][:])
    H = np.array(
        -2 * np.identity(n_basis)
        + np.diag(np.ones(n_basis - 1), 1)
        + np.diag(np.ones(n_basis - 1), -1)
    )

    LambdaLeft = 0.00001
    LambdaRight = 8
    # tol = 0.0001

    # Function (+ delta) and derivative f at left point
    G_left = getG(basis, H, identity, LambdaLeft, signal)
    G_leftDiff = getG(basis, H, identity, LambdaLeft + tol, signal)
    f_left = (G_leftDiff - G_left) / tol

    i = 0
    midpoint = LambdaLeft
    while abs(LambdaRight - LambdaLeft) > tol:
        midpoint = (LambdaRight + LambdaLeft) / 2
        # Function (+ delta) and derivative f at middle point
        G_middle = getG(basis, H, identity, midpoint, signal)
        G_middleDiff = getG(basis, H, identity, midpoint + tol, signal)
        f_middle = (G_middleDiff - G_middle) / tol

        if i > 100:
            print("Original choice of Lambda might not bracket minimum.")
            break

        # Continue with logic
        if f_left * f_middle > 0:
            # Throw away left half
            LambdaLeft = midpoint
            f_left = f_middle
        else:
            # Throw away right half
            LambdaRight = midpoint
        i = +1

    # NNLS fit of found minimum
    Lambda = midpoint
    s = nnls_fit(basis, H, Lambda, signal)

    # Determine chi2_min
    [_, resnormMin] = nnls(basis, signal)

    # Determine chi2_smooth
    y_recon = np.matmul(basis, s)
    resid = signal - y_recon
    resnormSmooth = np.sum(np.multiply(resid, resid))
    chi = resnormSmooth / resnormMin

    return s, chi, resid
