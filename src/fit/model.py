import numpy as np
import time

from scipy.optimize import curve_fit, nnls, least_squares, minimize
from src.fit.NNLS_reg_CV import NNLS_reg_CV
from symfit import parameters, variables, Fit, Parameter, exp, Ge


class Model(object):
    """Contains fitting methods of all different models."""

    class NNLS(object):
        @staticmethod
        def fit(
                idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int | None = 200
        ) -> tuple:
            """Standard fit for plain and regularised NNLS fitting."""
            try:
                fit, _ = nnls(basis, signal, maxiter=max_iter)
            except (RuntimeError, ValueError):
                fit = np.zeros(basis.shape[1])
            return idx, fit

        @staticmethod
        def model(b_values: np.ndarray, spectrum: np.ndarray, bins: np.ndarray):
            """Model to create fitted diffusion decay."""
            signal = 0
            for comp, d in enumerate(bins):
                signal += spectrum[comp] * np.exp(b_values * -d)
            return signal

    class NNLSregCV(object):
        @staticmethod
        def fit(
                idx: int, signal: np.ndarray, basis: np.ndarray, tol: float | None = 0.0001
        ) -> tuple:
            """Advanced NNLS fit including CV regularisation."""
            try:
                fit, _, _ = NNLS_reg_CV(basis, signal, tol)
            except (RuntimeError, ValueError):
                fit = np.zeros(basis.shape[1])
            return idx, fit

    class IVIM(object):
        @staticmethod
        def wrapper(n_components: int, TM: int):
            """
            Creates function for IVIM model, able to fill with partial.

            Boundaries should be organized in a recommended way.
                x0[0:n_components]: D values, slow to fast
                x0[n_components:-1]: f values, slow to (fast - 1). The fastest components fraction is calculated.
                x0[-1]: S0

            """

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
                        * (1 - (np.sum(args[n_components:-1])))
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
                args: np.ndarray,
                lb: np.ndarray,
                ub: np.ndarray,
                b_values: np.ndarray,
                n_components: int,
                max_iter: int,
                TM: int,
                timer: bool | None = False,
                **kwargs
        ):
            """Standard IVIM fit using the IVIM model wrapper."""
            start_time = time.time()

            try:
                fit_result = curve_fit(
                    Model.IVIM.wrapper(n_components=n_components, TM=TM),
                    b_values,
                    signal,
                    p0=args,
                    bounds=(lb, ub),
                    max_nfev=max_iter,
                    method="trf"
                )[0]
                if timer:
                    print(time.time() - start_time)
            except (RuntimeError, ValueError):
                fit_result = np.zeros(args.shape)
                if timer:
                    print("Error")
            return idx, fit_result

        @staticmethod
        def constraint_fractions(n_components: int) -> dict:
            """Constrains for fractions of IVIM fitting."""

            def func(args) -> dict:
                return 1 - sum(args[n_components:-1])

            return {"type": "ineq", "fun": func}

        @staticmethod
        def printer(n_components: int, args):
            """Model printer for testing."""
            f = f""
            for i in range(n_components - 1):
                f += f"exp(-kron(b_values, abs({args[i]}))) * {args[n_components + i]} + "
            f += f"exp(-kron(b_values, abs({args[n_components - 1]}))) * (1 - (sum({args[n_components:-1]})))"
            return f"( " + f + f" ) * {args[-1]}"

    class IVIMConstraint(object):
        @staticmethod
        def fit(idx, ydata, x0, lb, ub, b_values, max_iter, n_components, TM, **kwargs):
            constraints = kwargs.get("constraints", dict())
            method = kwargs.get("method", "trust-constr")
            try:
                fit_results = minimize(
                    fun=Model.IVIM.wrapper(n_components=n_components, TM=TM),
                    x0=x0,
                    args=ydata,
                    method=method,
                    bounds=(lb, ub),
                    constraints=constraints
                )
            except (RuntimeError, ValueError):
                pass

    class IVIMConstraintSymFit(object):
        @staticmethod
        def fit(
                idx: int,
                signal: np.ndarray,
                args: np.ndarray,

                lb: np.ndarray,
                ub: np.ndarray,
                b_values: np.ndarray,
                n_components: int,
                max_iter: int,
                TM: int,
                timer: bool | None = False,
                **kwargs
        ):
            d1 = Parameter("d1", value=args[0], min=lb[0], max=ub[0])
            d2 = Parameter("d2", value=args[1], min=lb[1], max=ub[1])
            d3 = Parameter("d3", value=args[2], min=lb[2], max=ub[2])
            # d1, d2, d3 = parameters("d1, d2, d3")
            f1 = Parameter("f1", value=args[3], min=lb[3], max=ub[3])
            f2 = Parameter("f2", value=args[4], min=lb[4], max=ub[4])
            # f1, f2 = parameters("f1, f2")
            s0 = Parameter("s0", value=args[5], min=lb[5], max=ub[5])
            # s0 = parameters("s0")
            x, y = variables("x, y")
            model = {s0 * (f1 * exp(- x * d1) + f2 * exp(- x * d2) + (
                    1 - f1 - f2) * exp(- x * d3))}
            fit = Fit(model, b_values, signal, constraints=[Ge(1 - f1 - f2, 0)])
            fit_results = fit.execute()
            fit_results_list = [fit_results.value(d1), fit_results.value(d2), fit_results.value(d3),
                                fit_results.value(f1), fit_results.value(f2), fit_results.value(s0)]
            return idx, fit_results_list
