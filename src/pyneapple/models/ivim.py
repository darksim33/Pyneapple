"""Multi-Exponential Models for fitting on the CPU.

This module provides the different multi-exponential models for fitting on the CPU.
For segmented fitting the last component is always used as fixed component and the
bounds have to be set accordingly.

Functions:
    mono_wrapper(**kwargs): Creates a mono-exponential model function.
    bi_wrapper(**kwargs): Creates a bi-exponential model function.
    tri_wrapper(**kwargs): Creates a tri-exponential model function.
    fit_curve(idx: int, signal: np.ndarray, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray,
        model: Callable, b_values: np.ndarray, max_iter: int, timer: bool | None = False,
        **kwargs): Standard exponential model fit using "curve_fit".

Note:
    Multiprocessing does not support dynamic parsing of methods. Therefore, the models
    are implemented explicitly and are selected by string comparison. Else this would
    cause pickling errors.

Version History:
    1.5.0 (2024-12-06):     Created the wrapper models. Added mono_wrapper, bi_wrapper,
                            tri_wrapper, and fit_curve functions. Removed IVIM and
                            IVIMFixedComponent classes.
"""

from __future__ import annotations

from collections.abc import Callable
import time
import numpy as np
from scipy.optimize import curve_fit


def mono_wrapper(**kwargs):
    """Creates a mono-exponential model function.

    Args:
        **kwargs: Additional keyword arguments.
            "reduced" (bool): Reduced model with only one component.
            "mixing_time" (float): Mixing time value. Needed for T1 fitting.

    """

    def model(b_values, *args):
        """Mono-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f/S0 , D, (mixing_time)) or
                (D, (mixing_time) for reduced.
        """
        f = 0
        if kwargs.get("reduced", False):
            f += np.exp(-np.kron(b_values, abs(args[0])))
        else:
            f += args[0] * np.exp(-np.kron(b_values, abs(args[1])))
        # add t1 fitting term
        if kwargs.get("mixing_time", None):
            f *= np.exp(-args[-1] / kwargs.get("TM", 1))
        return f

    return model


def bi_wrapper(**kwargs):
    """Creates a bi-exponential model function.

    There are different variants of the classic model available (see Models).

    Args:
        **kwargs: Additional keyword arguments.
            "reduced" (bool): Reduced model with only one component.
            "mixing_time" (float, None): Mixing time value. Needed for T1 fitting.
            "fixed_d" (float, None): Fixed D value for the second component.
    Models:
        f       = f1 * exp(-D1 * b) + f2 * exp(-D2 * b)
        f_t1    = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) * exp(-t1 / mixing_time)
            with: args[0] = f1, args[1] = D1, args[2] = f2, args[3] = D2,
            (args[4] = mixing_time)
        f_red   = f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b)
        f_red_t1= f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b) * exp(-t1 / mixing_time)
            with: args[0] = f1, args[1] = D1, args[2] = D2, (args[3] = mixing_time)
    """

    def model(b_values, *args):
        """Bi-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f1, D1, f2, D2, (mixing_time)) or
                (f1, D1, D2, (mixing_time)) for reduced.
        """
        # Add fist component f1*exp(-D1*b)
        f = args[0] * np.exp(-np.kron(b_values, abs(args[1])))
        # Add second component f
        if kwargs.get("reduced", False):  # (1-f1)*exp(-D2*b)
            if kwargs.get("fixed_d", None):
                f += (1 - args[0]) * np.exp(
                    -np.kron(b_values, abs(kwargs.get("fixed_d", 0)))
                )
            else:
                f += (1 - args[0]) * np.exp(-np.kron(b_values, abs(args[2])))
        else:  # f2*exp(-D2*b)
            if kwargs.get("fixed_d", None):
                f += args[2] * np.exp(-np.kron(b_values, abs(kwargs.get("fixed_d"))))
            else:
                f += args[2] * np.exp(-np.kron(b_values, abs(args[3])))
        # Add t1 fitting term
        if kwargs.get("mixing_time", None):  # * exp(-t1/mixing_time)
            f *= np.exp(-args[-1] / kwargs.get("mixing_time"))
        return f

    return model


def tri_wrapper(**kwargs):
    """Creates a tri-exponential model function.

    Args:
        **kwargs: Additional keyword arguments.
            "reduced" (bool): Reduced model with only one component.
            "mixing_time" (float, None): Mixing time value. Needed for T1 fitting.
            "fixed_d" (float, None): Fixed D value for the third component.
    Models:
        f       = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + f3 * exp(-D3 * b)
        f_t1    = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + f3 * exp(-D3 * b)
            * exp(-t1 / mixing_time) with: args[0] = f1, args[1] = D1, args[2] = f2,
            args[3] = D2, args[4] = f3, args[5] = D3, (args[6] = mixing_time)
        f_red   = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b)
        f_red_t1= f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b)
            * exp(-t1 / mixing_time) with: args[0] = f1, args[1] = D1, args[2] = f2,
            args[3] = D2, args[4] = D3, (args[5] = mixing_time)
    """

    def model(b_values, *args):
        """Tri-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f1 , D1, f2, D2, f3, D3, (mixing_time)) or
                    (f1, D1, f2, D2, D3, (mixing_time)) for reduced.
        """
        # Add first and second component f1*exp(-D1*b) + f2*exp(-D2*b)
        f = args[0] * np.exp(-np.kron(b_values, abs(args[1]))) + args[2] * np.exp(
            -np.kron(b_values, abs(args[3]))
        )
        if kwargs.get("reduce", False):  # (1-f1-f2)*exp(-D3*b)
            if kwargs.get("fixed_d", None):
                f += (1 - args[0] - args[2]) * np.exp(
                    -np.kron(b_values, abs(kwargs.get("fixed_d", 0)))
                )
            else:
                f += (1 - args[0] - args[2]) * np.exp(-np.kron(b_values, abs(args[4])))
        else:
            if kwargs.get("fixed_d", None):
                f += args[4] * np.exp(-np.kron(b_values, abs(kwargs.get("fixed_d"))))
            else:
                f += args[4] * np.exp(-np.kron(b_values, abs(args[5])))
        # Add t1 fitting term
        if kwargs.get("mixing_time", None):  # * exp(-t1/mixing_time)
            f *= np.exp(-args[-1] / kwargs.get("mixing_time"))
        return f

    return model


def get_model(model: str):
    if "mono" in model.lower():
        return mono_wrapper
    elif "bi" in model.lower():
        return bi_wrapper
    elif "tri" in model.lower():
        return tri_wrapper
    else:
        raise ValueError("Invalid model for fitting.")


def fit_curve(
    idx: int,
    signal: np.ndarray,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    model: str,
    b_values: np.ndarray,
    max_iter: int,
    timer: bool | None = False,
    **kwargs,
):
    """Standard exponential model fit using "curve_fit".

    Args:
        idx (int): Index of the voxel.
        signal (np.ndarray): Signal data.
        x0 (np.ndarray): Initial guess for the fit.
        lb (np.ndarray): Lower bounds for the fit.
        ub (np.ndarray): Upper bounds for the fit.
        model (str): Model name.
        b_values (np.ndarray): B-values of the signal.
        max_iter (int): Maximum number of iterations.
        timer (bool): Timer for the fit.
        **kwargs: Additional keyword arguments.
            reduced (bool): Reduced model for S/S0 fitting replacing one fraction
                (sum(f)=1).
            mixing_time (float): Mixing time value. Needed for T1 fitting.
    Returns:
        idx (int): Index of the voxel.
        fit_result (np.ndarray): Fit result holding only estimated parameters.
    """
    if timer:
        start_time = time.time()
    fit_model = get_model(model)
    try:
        fit_result = curve_fit(
            fit_model(**kwargs),
            b_values,
            signal,
            p0=x0,
            bounds=(lb, ub),
            max_nfev=max_iter,
            method=kwargs.get("algorithm", "trf"),
        )
    except (RuntimeError, ValueError):
        fit_result = np.zeros(x0.shape)
    if timer:
        print(time.time() - start_time)
    return idx, fit_result[0]


def fit_curve_fixed(
    idx: int,
    signal: np.ndarray,
    fixed_d: float | np.ndarray,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    model: str,
    b_values: np.ndarray,
    max_iter: int,
    timer: bool | None = False,
    **kwargs,
):
    """Standard exponential model fit using "curve_fit".

    Args:
        idx (int): Index of the voxel.
        signal (np.ndarray): Signal data.
        fixed_d (float | np.ndarray): Fixed D value for the last component.
        x0 (np.ndarray): Initial guess for the fit.
        lb (np.ndarray): Lower bounds for the fit.
        ub (np.ndarray): Upper bounds for the fit.
        model (str): Model name.
        b_values (np.ndarray): B-values of the signal.
        max_iter (int): Maximum number of iterations.
        timer (bool): Timer for the fit.
        **kwargs: Additional keyword arguments.
            reduced (bool): Reduced model for S/S0 fitting replacing one fraction
                (sum(f)=1).
            mixing_time (float): Mixing time value. Needed for T1 fitting.
    Returns:
        idx (int): Index of the voxel.
        fit_result (np.ndarray): Fit result holding only estimated parameters.
    """
    if timer:
        start_time = time.time()
    fit_model = get_model(model)
    try:
        fit_result = curve_fit(
            fit_model(fixed_d=fixed_d, **kwargs),
            b_values,
            signal,
            p0=x0,
            bounds=(lb, ub),
            max_nfev=max_iter,
            method=kwargs.get("algorithm", "trf"),
        )
    except (RuntimeError, ValueError):
        fit_result = np.zeros(x0.shape)
    if timer:
        print(time.time() - start_time)
    return idx, fit_result[0]


# class IVIM(object):
#     """Class for creating and fitting IVIM models.
#
#     Methods:
#         wrapper(n_components: int, **kwargs): Creates function for IVIM model,
#             able to fill with partial.
#         fit(idx: int, signal: np.ndarray, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray,
#             b_values: np.ndarray, n_components: int, max_iter: int,
#             timer: bool | None = False, **kwargs): Standard IVIM fit using the IVIM
#             model wrapper.
#     """
#
#     @staticmethod
#     def wrapper(n_components: int, **kwargs):
#         """Creates function for IVIM model, able to fill with partial. Returns
#             a function containing the IVIM model for a given number of components.
#
#         Boundaries should be organized in a recommended way.
#             x0[0:n_components]: D values, slow to fast
#             x0[n_components:2*n_components-1]: f values, slow to (fast - 1). The fastest components fraction is calculated.
#             x0[2*n_components-1]: S0
#             x0[-1]: T1
#
#         Args:
#             n_components (int): Number of components in the model.
#             **kwargs: Additional keyword arguments.
#
#         Returns:
#             multi_exp_model (Callable): IVIM model function
#         """
#
#         def multi_exp_model(b_values, *args):
#             """IVIM model function.
#
#             Actual IVIM model function. Returns the signal for a given set of B-values
#
#             Args:
#                 b_values (np.ndarray): B-values.
#                 *args: Arguments.
#             Returns:
#                 f (np.ndarray): Signal constructor for given B-values.
#             """
#             f = 0
#             for i in range(n_components - 1):
#                 f += np.exp(-np.kron(b_values, abs(args[i]))) * args[n_components + i]
#             f += (
#                 np.exp(-np.kron(b_values, abs(args[n_components - 1])))
#                 # Second half containing f, except for S0 as the very last entry
#                 * (1 - (np.sum(args[n_components : (2 * n_components - 1)])))
#             )
#
#             if kwargs.get("TM", None):
#                 # With nth entry being T1 in cases of T1 fitting
#                 # might not work at all
#                 f *= np.exp(-args[-1] / kwargs.get("TM"))
#
#             if kwargs.get("scale_image", None) == "S/S0":
#                 pass
#             else:
#                 f *= args[2 * n_components - 1]
#
#             return f  # Add S0 term for non-normalized signal
#
#         return multi_exp_model
#
#     @staticmethod
#     def fit(
#         idx: int,
#         signal: np.ndarray,
#         x0: np.ndarray,
#         lb: np.ndarray,
#         ub: np.ndarray,
#         b_values: np.ndarray,
#         n_components: int,
#         max_iter: int,
#         timer: bool | None = False,
#         **kwargs,
#     ):
#         """Standard IVIM fit using the IVIM model wrapper.
#
#         Args:
#             idx (int): Index of the voxel.
#             signal (np.ndarray): Signal data.
#             x0 (np.ndarray): Initial guess for the fit.
#             lb (np.ndarray): Lower bounds for the fit.
#             ub (np.ndarray): Upper bounds for the fit.
#             b_values (np.ndarray): B-values of the signal.
#             n_components (int): Number of components in the model.
#             max_iter (int): Maximum number of iterations.
#             timer (bool): Timer for the fit.
#             **kwargs: Additional keyword arguments.
#         Returns:
#             idx (int): Index of the voxel.
#             fit_result (np.ndarray): Fit result holding only estimated parameters.
#         """
#         start_time = time.time()
#
#         try:
#             fit_result = curve_fit(
#                 IVIM.wrapper(
#                     n_components=n_components,
#                     TM=kwargs.get("TM", None),
#                     scale_image=kwargs.get("scale_image", None),
#                 ),
#                 b_values,
#                 signal,
#                 p0=x0,
#                 bounds=(lb, ub),
#                 max_nfev=max_iter,
#                 method="trf",
#             )[0]
#             if timer:
#                 print(time.time() - start_time)
#         except (RuntimeError, ValueError):
#             fit_result = np.zeros(x0.shape)
#             if timer:
#                 print("Error")
#         return idx, fit_result
#
#     @staticmethod
#     def printer(n_components: int, args):
#         """Model printer for testing."""
#         f = ""
#         for i in range(n_components - 1):
#             f += f"exp(-kron(b_values, abs({args[i]}))) * {args[n_components + i]} + "
#         f += f"exp(-kron(b_values, abs({args[n_components - 1]}))) * (1 - (sum({args[n_components:-1]})))"
#         return "( " + f + f" ) * {args[-1]}"


# class IVIMFixedComponent(object):
#     """Class for creating and fitting IVIM models with fixed components.
#
#     In contrast to the classic IVIM model one component is fixed to a given value
#     determined in a prior fitting step.
#
#     Methods:
#         wrapper(n_components: int, **kwargs): Creates function for IVIM model,
#             able to fill with partial.
#         fit(idx: int, signal: np.ndarray, D_fixed: np.ndarray,
#             t1: np.ndarray | None = None, x0: np.ndarray = None, lb: np.ndarray = None,
#             ub: np.ndarray = None, b_values: np.ndarray = None, n_components: int = None,
#             max_iter: int = None, timer: bool | None = False, **kwargs):
#             IVIM fit using the IVIM model wrapper with fixed components.
#     """
#
#     @staticmethod
#     def wrapper(n_components: int, **kwargs):
#         """Creates function for IVIM model, able to fill with partial. Returns
#         a function containing the IVIM model for a given number of components.
#
#         Args:
#             n_components (int): Number of components in the model.
#             **kwargs: Additional keyword arguments.
#
#         Returns:
#             multi_exp_model (Callable): IVIM model function
#         """
#
#         def multi_exp_model(b_values: np.ndarray, *args):
#             """IVIM model function.
#
#             Args:
#                 b_values (np.ndarray): B-values.
#                 *args: Arguments.
#             Returns:
#                 f (np.ndarray): Signal constructor for given B-values.
#             """
#
#             f = 0
#             if n_components == 2:
#                 # args = [D_fast, f_fast, S0, (T1)]
#                 # kwargs["D_fixed", "t1"] = D_slow, t1
#                 f += np.exp(-np.kron(b_values, args[0])) * args[1]  # D_fast term
#                 f += (1 - args[1]) * np.exp(
#                     -np.kron(b_values, kwargs.get("D_fixed", 0))
#                 )  # D_slow term
#
#             elif n_components == 3:
#                 # args = [D_interm, D_fast, f_interm,  f_fast, S0, (T1)]
#                 # kwargs["D_slow", "t1"] = D_slow, t1
#                 f += (
#                     np.exp(-np.kron(b_values, args[0])) * args[2]
#                     + np.exp(-np.kron(b_values, args[1])) * args[3]
#                     + (1 - args[1])
#                     * np.exp(
#                         -np.kron(b_values, kwargs.get("D_fixed", 0))
#                     )  # D_slow term
#                 )
#             if kwargs.get("TM", None):
#                 # if there is a t1 value deployed by kwargs t1 will not be fitted instead this term will work
#                 # as correction term
#                 f += np.exp(-kwargs.get("t1", args[-1]) / kwargs.get("TM"))
#
#                 if not kwargs.get("scale_image", None) == "S/S0":
#                     f *= args[-2]
#             else:
#                 if not kwargs.get("scale_image", None) == "S/S0":
#                     f *= args[-1]
#
#             return f
#
#         return multi_exp_model
#
#     @staticmethod
#     def fit(
#         idx: int,
#         signal: np.ndarray,
#         D_fixed: np.ndarray,
#         t1: np.ndarray | None = None,
#         x0: np.ndarray = None,
#         lb: np.ndarray = None,
#         ub: np.ndarray = None,
#         b_values: np.ndarray = None,
#         n_components: int = None,
#         max_iter: int = None,
#         timer: bool | None = False,
#         **kwargs,
#     ):
#         """IVIM fit using the IVIM model wrapper with fixed components.
#
#         Args:
#             idx (int): Index of the voxel.
#             signal (np.ndarray): Signal data.
#             D_fixed (np.ndarray): Fixed D value.
#             t1 (np.ndarray): Fixed T1 value.
#             x0 (np.ndarray): Initial guess for the fit.
#             lb (np.ndarray): Lower bounds for the fit.
#             ub (np.ndarray): Upper bounds for the fit.
#             b_values (np.ndarray): B-values of the signal.
#             n_components (int): Number of components in the model.
#             max_iter (int): Maximum number of iterations.
#             timer (bool): Timer for the fit.
#             **kwargs: Additional keyword arguments.
#         Returns:
#             idx (int): Index of the voxel.
#             fit_result (np.ndarray): Fit result holding only estimated parameters.
#         """
#         start_time = time.time()
#         try:
#             fit_result = curve_fit(
#                 IVIMFixedComponent.wrapper(
#                     n_components=n_components,
#                     TM=kwargs.get("TM", None),
#                     scale_image=kwargs.get("scale_image", None),
#                     D_fixed=D_fixed,
#                     t1=t1,
#                 ),
#                 b_values,
#                 signal,
#                 p0=x0,
#                 bounds=(lb, ub),
#                 max_nfev=max_iter,
#                 method="trf",
#             )[0]
#             if timer:
#                 print(time.time() - start_time)
#         except (RuntimeError, ValueError):
#             fit_result = np.zeros(x0.shape)
#             if timer:
#                 print("Error")
#         return idx, fit_result
