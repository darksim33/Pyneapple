import numpy as np
import time

from scipy.optimize import curve_fit, nnls
from .NNLS_reg_CV import NNLS_reg_CV


# from symfit import parameters, variables, Fit, Parameter, exp, Ge


class Model(object):
    """Contains fitting methods of all different models."""

    class NNLS(object):
        @staticmethod
        def fit(
            idx: int | tuple,
            signal: np.ndarray,
            basis: np.ndarray,
            max_iter: int | None,
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

    class NNLSCV(object):
        @staticmethod
        def fit(
            idx: int,
            signal: np.ndarray,
            basis: np.ndarray,
            tol: float | None,
            max_iter: int | None,
        ) -> tuple:
            """Advanced NNLS fit including CV regularisation."""
            try:
                fit, _, _ = NNLS_reg_CV(basis, signal, tol, max_iter)
            except (RuntimeError, ValueError):
                fit = np.zeros(basis.shape[1])
            return idx, fit

    class IVIM(object):
        @staticmethod
        def wrapper(n_components: int, **kwargs):
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

                if kwargs.get("TM", None):
                    # With nth entry being T1 in cases of T1 fitting
                    # might not work at all
                    f *= np.exp(-args[n_components] / kwargs.get("TM"))

                if not kwargs.get("scale_image", None) == "S/S0":
                    f *= args[-1]

                return f  # Add S0 term for non-normalized signal

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
            timer: bool | None = False,
            **kwargs,
        ):
            """Standard IVIM fit using the IVIM model wrapper."""
            start_time = time.time()

            try:
                fit_result = curve_fit(
                    Model.IVIM.wrapper(
                        n_components=n_components,
                        TM=kwargs.get("TM", None),
                        scale_image=kwargs.get("scale_image", None),
                    ),
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

    class IVIMFixedComponent(object):
        @staticmethod
        def wrapper(n_components: int, **kwargs):
            def multi_exp_model(b_values, *args):
                f = 0
                if n_components == 2:
                    # args = [D_fast, f_fast, S0, (T1)]
                    # kwargs["D_slow", "t1"] = D_slow, t1
                    f += np.exp(-np.kron(b_values, args[0])) * args[1]  # D_fast term
                    f += (1 - args[1]) * np.exp(
                        -np.kron(b_values, kwargs.get("D_fixed", 0))
                    )  # D_slow term

                    if kwargs.get("TM", None):
                        # if there is a t1 value deployed by kwargs t1 will not be fitted instead this term will work
                        # as correction term
                        f += np.exp(-kwargs.get("t1", args[-1]) / kwargs.get("TM"))

                        if not kwargs.get("scale_image", None) == "S/S0":
                            f *= agrs[-2]
                    else:
                        if not kwargs.get("scale_image", None) == "S/S0":
                            f *= agrs[-1]
                return f

            return multi_exp_model

        @staticmethod
        def fit(
            idx: int,
            signal: np.ndarray,
            D_fixed: np.ndarray,
            t1: np.ndarray | None = None,
            x0: np.ndarray = None,
            lb: np.ndarray = None,
            ub: np.ndarray = None,
            b_values: np.ndarray = None,
            n_components: int = None,
            max_iter: int = None,
            timer: bool | None = False,
            **kwargs,
        ):
            start_time = time.time()
            try:
                fit_result = curve_fit(
                    Model.IVIMFixedComponent.wrapper(
                        n_components=n_components,
                        TM=kwargs.get("TM", None),
                        scale_image=kwargs.get("scale_image", None),
                        D_fixed=D_fixed,
                        t1=t1,
                    ),
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
