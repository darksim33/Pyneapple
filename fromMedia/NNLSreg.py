import numpy as np
from scipy.optimize import nnls
from scipy.linalg import norm

# renamed file and main function
# added maxiter to nnls


def nnlsfit(A, H, Lambda, signal):
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


def getG(A, H, I, Lambda, signal):
    # Determining lambda function G

    NNLSfit = nnlsfit(A, H, Lambda, signal)
    # Calculating G with CrossValidation method
    G = (
        norm(signal - np.matmul(A, NNLSfit)) ** 2
        / np.trace(
            I
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


def NNLSreg(DBasis, signal):
    # Regularised NNLS fitting based on CVNNLS.m of the AnalyzeNNLS by Bjarnason et al.

    # Identity matrix
    I = np.identity(len(signal))

    # Curvature
    Dlength = len(DBasis[1][:])
    H = np.array(
        -2 * np.identity(Dlength)
        + np.diag(np.ones(Dlength - 1), 1)
        + np.diag(np.ones(Dlength - 1), -1)
    )

    LambdaLeft = 0.00001
    LambdaRight = 8
    tol = 0.0001

    # Function (+ delta) and derivative f at left point
    G_left = getG(DBasis, H, I, LambdaLeft, signal)
    G_leftDiff = getG(DBasis, H, I, LambdaLeft + tol, signal)
    f_left = (G_leftDiff - G_left) / tol

    i = 0
    while abs(LambdaRight - LambdaLeft) > tol:
        midpoint = (LambdaRight + LambdaLeft) / 2
        # Function (+ delta) and derivative f at middle point
        G_middle = getG(DBasis, H, I, midpoint, signal)
        G_middleDiff = getG(DBasis, H, I, midpoint + tol, signal)
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
    s = nnlsfit(DBasis, H, Lambda, signal)

    # Determin chi2_min
    [amp_min, resnormMin] = nnls(DBasis, signal)

    # Determin chi2_smooth
    y_recon = np.matmul(DBasis, s)
    resid = signal - y_recon
    resnormSmooth = np.sum(np.multiply(resid, resid))
    chi = resnormSmooth / resnormMin

    return s, chi, resid
