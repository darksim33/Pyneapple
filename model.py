import numpy as np
from scipy.optimize import least_squares, curve_fit, nnls
from fitting.NNLSregCV import NNLSregCV


class Model(object):
    def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int = 200):
        """NNLS fitting model (may include regularisation)"""

        fit, _ = nnls(basis, signal, maxiter=max_iter)
        return idx, fit

    def NNLS_reg_CV(
        idx: int, signal: np.ndarray, basis: np.ndarray, tol: float = 0.0001
    ):
        """NNLS fitting model with cross-validation algorithm for automatic regularisation weighting"""

        fit, _, _ = NNLSregCV(basis, signal, tol)
        return idx, fit

    def mono(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        max_iter: int,
        TM: float | None,
    ):
        """Mono exponential fitting model for ADC and T1"""
        # NOTE does not theme to work at all

        def mono_wrapper(TM: float | None):
            def mono_model(
                b_values: np.ndarray,
                S0: float | int,
                x0: float | int,
                T1: float | int = 0,
            ):
                if TM is None or 0:
                    return np.array(S0 * np.exp(-np.kron(b_values, x0)))

                return np.array(S0 * np.exp(-np.kron(b_values, x0)) * np.exp(-T1 / TM))

            return mono_model

        fit, _ = curve_fit(
            mono_wrapper(TM),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )
        return idx, fit

    def multi_exp(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_components: int,
        max_iter: int,
    ):
        """Multiexponential fitting model (e.g. for NLLS, mono, IDEAL ...)"""

        def multi_exp_wrapper(n_components: int):
            def multi_exp_model(b_values: np.ndarray, x0: float | int):
                f = 0
                for i in range(n_components - 2):
                    f = +np.exp(-np.kron(b_values, abs(x0[i]))) * x0[n_components + i]
                return f + np.exp(-np.kron(b_values, abs(x0[n_components - 1]))) * (
                    100 - (np.sum(x0[n_components:]))
                )

            return multi_exp_model

        fit, _ = curve_fit(
            multi_exp_wrapper(n_components=n_components),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit
