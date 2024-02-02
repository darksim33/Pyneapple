import numpy as np
import time

from scipy.optimize import curve_fit, nnls, least_squares, minimize, LinearConstraint
from src.fit.NNLS_reg_CV import NNLS_reg_CV

# from symfit import parameters, variables, Fit, Parameter, exp, Ge


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
            x0: np.ndarray,
            lb: np.ndarray,
            ub: np.ndarray,
            b_values: np.ndarray,
            n_components: int,
            max_iter: int,
            TM: int,
            timer: bool | None = False,
            **kwargs,
        ):
            """Standard IVIM fit using the IVIM model wrapper."""
            start_time = time.time()

            try:
                fit_result = curve_fit(
                    Model.IVIM.wrapper(n_components=n_components, TM=TM),
                    b_values,
                    signal,
                    p0=x0,
                    bounds=(lb, ub),
                    max_nfev=max_iter,
                    method="trf",
                )[0]
                if timer:
                    print(time.time() - start_time)
            except (RuntimeError, ValueError):
                fit_result = np.zeros(x0.shape)
                if timer:
                    print("Error")
            return idx, fit_result

        @staticmethod
        def printer(n_components: int, args):
            """Model printer for testing."""
            f = f""
            for i in range(n_components - 1):
                f += f"exp(-kron(b_values, abs({args[i]}))) * {args[n_components + i]} + "
            f += f"exp(-kron(b_values, abs({args[n_components - 1]}))) * (1 - (sum({args[n_components:-1]})))"
            return f"( " + f + f" ) * {args[-1]}"

    class IVIMReduced(object):
        @staticmethod
        def wrapper(n_components: int):
            def model(xdata, *args):
                f = 0
                for i in range(n_components - 1):
                    f += np.exp(-np.kron(xdata, abs(args[i]))) * args[n_components + i]
                f += (
                    np.exp(-np.kron(xdata, abs(args[n_components - 1])))
                    # Second half containing f, except for S0 as the very last entry
                    * (1 - (np.sum(args[n_components:])))
                )

                return f  # Add S0 term for non-normalized signal

            return model

        @staticmethod
        def fit(idx, ydata, x0, lb, ub, b_values, n_components, max_iter, **kwargs):
            """Standard IVIM fit using the IVIM model wrapper for S/S0."""
            try:
                fit_result = curve_fit(
                    Model.IVIMReduced.wrapper(n_components=n_components),
                    b_values,
                    ydata,
                    p0=x0,
                    bounds=(lb, ub),
                    max_nfev=max_iter,
                    method="trf",
                )[0]
            except (RuntimeError, ValueError):
                fit_result = np.zeros(x0.shape)
            return idx, fit_result

    # class IVIMConstraintSymFit(object):
    #     @staticmethod
    #     def fit(
    #         idx: int,
    #         signal: np.ndarray,
    #         args: np.ndarray,
    #         lb: np.ndarray,
    #         ub: np.ndarray,
    #         b_values: np.ndarray,
    #         n_components: int,
    #         max_iter: int,
    #         TM: int,
    #         timer: bool | None = False,
    #         **kwargs,
    #     ):
    #         d1 = Parameter("d1", value=args[0], min=lb[0], max=ub[0])
    #         d2 = Parameter("d2", value=args[1], min=lb[1], max=ub[1])
    #         d3 = Parameter("d3", value=args[2], min=lb[2], max=ub[2])
    #         # d1, d2, d3 = parameters("d1, d2, d3")
    #         f1 = Parameter("f1", value=args[3], min=lb[3], max=ub[3])
    #         f2 = Parameter("f2", value=args[4], min=lb[4], max=ub[4])
    #         # f1, f2 = parameters("f1, f2")
    #         s0 = Parameter("s0", value=args[5], min=lb[5], max=ub[5])
    #         # s0 = parameters("s0")
    #         x, y = variables("x, y")
    #         model = {
    #             s0
    #             * (f1 * exp(-x * d1) + f2 * exp(-x * d2) + (1 - f1 - f2) * exp(-x * d3))
    #         }
    #         fit = Fit(model, b_values, signal, constraints=[Ge(1 - f1 - f2, 0)])
    #         fit_results = fit.execute()
    #         fit_results_list = [
    #             fit_results.value(d1),
    #             fit_results.value(d2),
    #             fit_results.value(d3),
    #             fit_results.value(f1),
    #             fit_results.value(f2),
    #             fit_results.value(s0),
    #         ]
    #         return idx, fit_results_list

    # class IVIMConstraint(object):
    #     @staticmethod
    #     def fit(
    #         idx: int,
    #         signal: np.ndarray,
    #         args: np.ndarray,
    #         lb: np.ndarray,
    #         ub: np.ndarray,
    #         b_values: np.ndarray,
    #         n_components: int,
    #         max_iter: int,
    #         TM: int,
    #         timer: bool | None = False,
    #         **kwargs,
    #     ):
    #         d1 = Parameter("d1", value=args[0], min=lb[0], max=ub[0])
    #         d2 = Parameter("d2", value=args[1], min=lb[1], max=ub[1])
    #         d3 = Parameter("d3", value=args[2], min=lb[2], max=ub[2])
    #         # d1, d2, d3 = parameters("d1, d2, d3")
    #         f1 = Parameter("f1", value=args[3], min=lb[3], max=ub[3])
    #         f2 = Parameter("f2", value=args[4], min=lb[4], max=ub[4])
    #         # f1, f2 = parameters("f1, f2")
    #         s0 = Parameter("s0", value=args[5], min=lb[5], max=ub[5])
    #         # s0 = parameters("s0")
    #         x, y = variables("x, y")
    #         model = {
    #             s0
    #             * (f1 * exp(-x * d1) + f2 * exp(-x * d2) + (1 - f1 - f2) * exp(-x * d3))
    #         }
    #         fit = Fit(model, b_values, signal, constraints=[Ge(1 - f1 - f2, 0)])
    #         fit_results = fit.execute()
    #         fit_results_list = [
    #             fit_results.value(d1),
    #             fit_results.value(d2),
    #             fit_results.value(d3),
    #             fit_results.value(f1),
    #             fit_results.value(f2),
    #             fit_results.value(s0),
    #         ]
    #         return idx, fit_results_list

    class IVIMCopilot(object):
        @staticmethod
        def model(x, *args):
            d1, d2, d3, f1, f2, s0 = args
            return s0 * (
                np.exp(-np.kron(x, abs(d1))) * f1
                + np.exp(-np.kron(x, abs(d2))) * f2
                + np.exp(-np.kron(x, abs(d3))) * (1 - f1 - f2)
            )

        @staticmethod
        def get_linear_constraints():
            """Define Linear Constraint 1 - a -b > 0"""
            return [LinearConstraint(A=[[0, 0, 0, -1, -1, 0]], lb=[0], ub=[1])]

        @staticmethod
        def get_boundaries(lbs: list, ubs: list) -> list[tuple]:
            """Define boundaries for minimize function."""
            bounds = list()
            for lb, ub in zip(lbs, ubs):
                bounds.append((lb, ub))
            return bounds

        @staticmethod
        def error_function(x: np.ndarray, y: np.ndarray, args: list) -> float:
            return np.mean(np.square(Model.IVIMCopilot.model(x, *args) - y))

        @staticmethod
        def fit(idx, signal, args, b_values, lb, ub, **kwargs):
            method = kwargs.get("method", "trust-constr")
            # constraints = kwargs.get(
            #     "constraints", Model.IVIMCopilot.get_linear_constraints()
            # )
            constraints = None
            try:
                print(idx)
                fit = minimize(
                    fun=lambda x: np.mean(
                        np.square(Model.IVIMCopilot.model(b_values, *x) - signal)
                    ),
                    x0=args,
                    bounds=Model.IVIMCopilot.get_boundaries(lb, ub),
                    method=method,
                    constraints=constraints,
                )
                return idx, fit.x
            except (TypeError, ValueError):
                return idx, np.zeros(args.shape)

    class IVIMCopilot2(object):
        @staticmethod
        def wrapper(xdata, ydata):
            def model(x, *args):
                d1, d2, d3, f1, f2, s0 = args
                return s0 * (
                    np.exp(-np.kron(x, abs(d1))) * f1
                    + np.exp(-np.kron(x, abs(d2))) * f2
                    + np.exp(-np.kron(x, abs(d3))) * (1 - f1 - f2)
                )

            return lambda var: np.mean(np.square(model(xdata, *var) - ydata))

        @staticmethod
        def fit(idx, signal, args, b_values, lb, ub, **kwargs):
            constraints = Model.IVIMCopilot.get_linear_constraints()
            bounds = Model.IVIMCopilot.get_boundaries(lb, ub)
            print(idx)
            try:
                fit = minimize(
                    fun=Model.IVIMCopilot2.wrapper(b_values, signal),
                    x0=args,
                    bounds=bounds,
                    constraints=constraints,
                )
                return idx, fit.x
            except (RuntimeError, TypeError, ValueError):
                return idx, np.zeros(args.shape)


"""
    import numpy as np
    from scipy.optimize import minimize

    # define a superposition of exponential terms as a function of x and p
    def f(x, *p):
        a, b, c, d, e = p
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e

    # generate some noisy data with known coefficients
    p0 = [2, 0.5, 1, 0.2, 0.1]
    x = np.linspace(0, 10, 100) 
    y = f(x, *p0)
    y_noise = y + np.random.randn(100) * 0.1

    # define the mean squared error as a function of the parameters
    err = lambda p: np.mean((f(x, *p) - y_noise) ** 2)

    # minimize the error using scipy.minimize with some bounds on the parameters
    p_init = [1, 1, 1, 1, 1]
    p_opt = minimize(
        err, # objective function to minimize
        p_init, # initial guess for the parameters
        bounds=[(0, None), (0, None), (0, None), (0, None), (None, None)], # lower and upper bounds for each parameter
        method="L-BFGS-B" # this method supports bounds
    ).x

    # print the optimized parameters and the true parameters
    print("Optimized parameters:", p_opt)
    print("True parameters:", p0)

"""
