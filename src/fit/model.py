import numpy as np
import time

from scipy.optimize import curve_fit, nnls
from src.fit.NNLS_reg_CV import NNLS_reg_CV


class Model(object):
    class NNLS(object):
        @staticmethod
        def fit(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int | None = 200) -> tuple:
            """NNLS fitting model (may include regularisation)"""

            fit, _ = nnls(basis, signal, maxiter=max_iter)
            return idx, fit

    class NNLSregCV(object):
        @staticmethod
        def fit(idx: int, signal: np.ndarray, basis: np.ndarray, tol: float | None = 0.1) -> tuple:
            fit, _, _ = NNLS_reg_CV(basis, signal, tol)
            return idx, fit

    class MultiExp(object):
        @staticmethod
        def wrapper(n_components: int, TM: int):
            def multi_exp_model(b_values, *args):
                f = 0
                for i in range(n_components - 1):
                    f += (
                            np.exp(-np.kron(b_values, abs(args[i])))
                            * args[n_components + i]
                    )
                f += (
                        np.exp(-np.kron(b_values, abs(args[n_components - 1])))
                        # Second half containing f, except for S0 as the very last entry
                        * (1 - (np.sum(args[n_components: -1])))
                )

                if TM:
                    # With nth entry being T1 in cases of T1 fitting
                    f *= np.exp(-args[n_components] / TM)

                return f * args[-1]  # Add S0 term for non-normalized signal

            return multi_exp_model

        @staticmethod
        def fit(
            idx: int,
            signal: np.ndarray,
            b_values: np.ndarray,
            n_components: int,
            args: np.ndarray,
            lb: np.ndarray,
            ub: np.ndarray,
            max_iter: int,
            TM: int,
            timer: bool | None = False,
        ):
            start_time = time.time()

            try:
                fit_result = curve_fit(
                    Model.MultiExp.wrapper(n_components=n_components, TM=TM),
                    b_values,
                    signal,
                    p0=args,
                    bounds=(lb, ub),
                    max_nfev=max_iter,
                )[0]
                if timer:
                    print(time.time() - start_time)
            except(RuntimeError, ValueError):
                fit_result = np.zeros(args.shape)
                if timer:
                    print("Error")
            return idx, fit_result

        @staticmethod
        def printer(n_components: int, args):
            f = f""
            for i in range(n_components - 1):
                f += f"exp(-kron(b_values, abs({args[i]}))) * {args[n_components + i]} + "
            f += f"exp(-kron(b_values, abs({args[n_components-1]}))) * (1 - (sum({args[n_components:-1]})))"
            return f"( " + f + f" ) * {args[-1]}"
