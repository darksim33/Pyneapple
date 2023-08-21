import numpy as np
from scipy.optimize import curve_fit, nnls
from .NNLSregCV import NNLSregCV

# from fit import FitData


class Model(object):
    """Model class returning fit of selected model with applied parameters"""

    @staticmethod
    def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int = 200):
        """NNLS fitting model (may include regularisation)"""

        fit, _ = nnls(basis, signal, maxiter=max_iter)
        return idx, fit

    @staticmethod
    def NNLS_reg_CV(
        idx: int, signal: np.ndarray, basis: np.ndarray, tol: float = 0.0001
    ):
        """NNLS fitting model with cross-validation algorithm for automatic regularisation weighting"""

        fit, _, _ = NNLSregCV(basis, signal, tol)
        return idx, fit

    @staticmethod
    def mono(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        args: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        max_iter: int,
        TM: float | None,
    ):
        """Mono exponential fitting model for ADC and T1"""
        # NOTE does not theme to work for T1

        def mono_wrapper(TM: float | None):
            # TODO: use multi_exp(n_components=1) etc.

            def mono_model(b_values: np.ndarray, *args):
                f = np.array(args[0] * np.exp(-np.kron(b_values, args[1]))) * args[-1]

                if TM is not None or not 0:
                    f = f * np.exp(-args[2] / TM)

                return f

            return mono_model

        fit = curve_fit(
            mono_wrapper(TM),
            b_values,
            signal,
            args,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )
        return idx, fit

    @staticmethod
    def multi_exp(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        args: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_components: int,
        max_iter: int,
    ):
        """Multi-exponential fitting model (e.g. for NLLS, mono, IDEAL ...)"""

        # TODO: working? testing needed
        def multi_exp_wrapper(n_components: int):
            def multi_exp_model(b_values, *args):
                f = 0
                for i in range(1, n_components):
                    f += (
                        np.exp(-np.kron(b_values, abs(args[i])))
                        * args[n_components + i]
                    )
                return (
                    (
                        f
                        + np.exp(-np.kron(b_values, abs(args[n_components])))
                        * (1 - (np.sum(args[-n_components:-1])))
                        # Last n_components entries containing f, but S0 as the last entry
                    )
                    * args[-1]
                    # S0 term for non normalized signal
                )

            return multi_exp_model

        # def multi_exp_printer(n_components: int):
        #     def multi_exp_model(b_values, x0):
        #         f = f"0 + "
        #         for i in range(n_components - 1):
        #             f += f"np.exp(-np.kron(b_values, abs(x0[{i}]))) * x0[{n_components} + {i}] + "
        #         f += f"np.exp(-np.kron(b_values, abs(x0[{n_components - 1}]))) * (100 - (np.sum(x0[n_components:])))"
        #         return f"( " + f + f" ) * x0({n_components} + {i} + {1})"
        #
        #     return multi_exp_model

        fit = curve_fit(
            multi_exp_wrapper(n_components),
            b_values,
            signal,
            args,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )
        return idx, fit
