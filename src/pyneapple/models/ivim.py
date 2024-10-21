"""IVIM model for fitting IVIM data.

This module provides the IVIM class for creating and fitting IVIM models.

Classes:
    IVIM: Class for creating and fitting IVIM models.
    IVIMFixedComponent: Class for creating and fitting IVIM models with fixed components.

Functions:
    IVIM.wrapper(n_components: int, **kwargs): Creates function for IVIM model,
        able to fill with partial.
    IVIM.fit(idx: int, signal: np.ndarray, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray,
        b_values: np.ndarray, n_components: int, max_iter: int,
        timer: bool | None = False, **kwargs): Standard IVIM fit using the IVIM
        model wrapper.
    IVIM.printer(n_components: int, args): Model printer for testing.
    IVIMFixedComponent.wrapper(n_components: int, **kwargs): Creates function for IVIM model,
        able to fill with partial.
    IVIMFixedComponent.fit(idx: int, signal: np.ndarray, D_fixed: np.ndarray,
        t1: np.ndarray | None = None, x0: np.ndarray = None, lb: np.ndarray = None,
        ub: np.ndarray = None, b_values: np.ndarray = None, n_components: int = None,
        max_iter: int = None, timer: bool | None = False, **kwargs):
        IVIM fit using the IVIM model wrapper with fixed components.
"""

from __future__ import annotations

import numpy as np
import time

from scipy.optimize import curve_fit


class IVIM(object):
    """Class for creating and fitting IVIM models.

    Methods:
        wrapper(n_components: int, **kwargs): Creates function for IVIM model,
            able to fill with partial.
        fit(idx: int, signal: np.ndarray, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray,
            b_values: np.ndarray, n_components: int, max_iter: int,
            timer: bool | None = False, **kwargs): Standard IVIM fit using the IVIM
            model wrapper.
    """

    @staticmethod
    def wrapper(n_components: int, **kwargs):
        """Creates function for IVIM model, able to fill with partial. Returns a
            a function containing the IVIM model for a given number of components.

        Boundaries should be organized in a recommended way.
            x0[0:n_components]: D values, slow to fast
            x0[n_components:2*n_components-1]: f values, slow to (fast - 1). The fastest components fraction is calculated.
            x0[2*n_components-1]: S0
            x0[-1]: T1

        Args:
            n_components (int): Number of components in the model.
            **kwargs: Additional keyword arguments.

        Returns:
            multi_exp_model (Callable): IVIM model function
        """

        def multi_exp_model(b_values, *args):
            """IVIM model function.

            Actual IVIM model function. Returns the signal for a given set of B-values

            Args:
                b_values (np.ndarray): B-values.
                *args: Arguments.
            Returns:
                f (np.ndarray): Signal constructor for given B-values.
            """
            f = 0
            for i in range(n_components - 1):
                f += np.exp(-np.kron(b_values, abs(args[i]))) * args[n_components + i]
            f += (
                np.exp(-np.kron(b_values, abs(args[n_components - 1])))
                # Second half containing f, except for S0 as the very last entry
                * (1 - (np.sum(args[n_components : (2 * n_components - 1)])))
            )

            if kwargs.get("TM", None):
                # With nth entry being T1 in cases of T1 fitting
                # might not work at all
                f *= np.exp(-args[-1] / kwargs.get("TM"))

            if not kwargs.get("scale_image", None) == "S/S0":
                f *= args[2 * n_components - 1]

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
        """Standard IVIM fit using the IVIM model wrapper.

        Args:
            idx (int): Index of the voxel.
            signal (np.ndarray): Signal data.
            x0 (np.ndarray): Initial guess for the fit.
            lb (np.ndarray): Lower bounds for the fit.
            ub (np.ndarray): Upper bounds for the fit.
            b_values (np.ndarray): B-values of the signal.
            n_components (int): Number of components in the model.
            max_iter (int): Maximum number of iterations.
            timer (bool): Timer for the fit.
            **kwargs: Additional keyword arguments.
        Returns:
            idx (int): Index of the voxel.
            fit_result (np.ndarray): Fit result holding only estimated parameters.
        """
        start_time = time.time()

        try:
            fit_result = curve_fit(
                IVIM.wrapper(
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
        f = ""
        for i in range(n_components - 1):
            f += f"exp(-kron(b_values, abs({args[i]}))) * {args[n_components + i]} + "
        f += f"exp(-kron(b_values, abs({args[n_components - 1]}))) * (1 - (sum({args[n_components:-1]})))"
        return "( " + f + f" ) * {args[-1]}"


class IVIMFixedComponent(object):
    """Class for creating and fitting IVIM models with fixed components.

    In contrast to the classic IVIM model one component is fixed to a given value
    determined in a prior fitting step.

    Methods:
        wrapper(n_components: int, **kwargs): Creates function for IVIM model,
            able to fill with partial.
        fit(idx: int, signal: np.ndarray, D_fixed: np.ndarray,
            t1: np.ndarray | None = None, x0: np.ndarray = None, lb: np.ndarray = None,
            ub: np.ndarray = None, b_values: np.ndarray = None, n_components: int = None,
            max_iter: int = None, timer: bool | None = False, **kwargs):
            IVIM fit using the IVIM model wrapper with fixed components.
    """

    @staticmethod
    def wrapper(n_components: int, **kwargs):
        """Creates function for IVIM model, able to fill with partial. Returns a
        a function containing the IVIM model for a given number of components.

        Args:
            n_components (int): Number of components in the model.
            **kwargs: Additional keyword arguments.

        Returns:
            multi_exp_model (Callable): IVIM model function
        """

        def multi_exp_model(b_values: np.ndarray, *args):
            """IVIM model function.

            Args:
                b_values (np.ndarray): B-values.
                *args: Arguments.
            Returns:
                f (np.ndarray): Signal constructor for given B-values.
            """

            f = 0
            if n_components == 2:
                # args = [D_fast, f_fast, S0, (T1)]
                # kwargs["D_fixed", "t1"] = D_slow, t1
                f += np.exp(-np.kron(b_values, args[0])) * args[1]  # D_fast term
                f += (1 - args[1]) * np.exp(
                    -np.kron(b_values, kwargs.get("D_fixed", 0))
                )  # D_slow term

            elif n_components == 3:
                # args = [D_interm, D_fast, f_interm,  f_fast, S0, (T1)]
                # kwargs["D_slow", "t1"] = D_slow, t1
                f += (
                    np.exp(-np.kron(b_values, args[0])) * args[2]
                    + np.exp(-np.kron(b_values, args[1])) * args[3]
                    + (1 - args[1])
                    * np.exp(
                        -np.kron(b_values, kwargs.get("D_fixed", 0))
                    )  # D_slow term
                )
            if kwargs.get("TM", None):
                # if there is a t1 value deployed by kwargs t1 will not be fitted instead this term will work
                # as correction term
                f += np.exp(-kwargs.get("t1", args[-1]) / kwargs.get("TM"))

                if not kwargs.get("scale_image", None) == "S/S0":
                    f *= args[-2]
            else:
                if not kwargs.get("scale_image", None) == "S/S0":
                    f *= args[-1]

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
        """IVIM fit using the IVIM model wrapper with fixed components.

        Args:
            idx (int): Index of the voxel.
            signal (np.ndarray): Signal data.
            D_fixed (np.ndarray): Fixed D value.
            t1 (np.ndarray): Fixed T1 value.
            x0 (np.ndarray): Initial guess for the fit.
            lb (np.ndarray): Lower bounds for the fit.
            ub (np.ndarray): Upper bounds for the fit.
            b_values (np.ndarray): B-values of the signal.
            n_components (int): Number of components in the model.
            max_iter (int): Maximum number of iterations.
            timer (bool): Timer for the fit.
            **kwargs: Additional keyword arguments.
        Returns:
            idx (int): Index of the voxel.
            fit_result (np.ndarray): Fit result holding only estimated parameters.
        """
        start_time = time.time()
        try:
            fit_result = curve_fit(
                IVIMFixedComponent.wrapper(
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
