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

import time
import numpy as np
from scipy.optimize import curve_fit
from ..utils.logger import logger
from .model import AbstractFitModel


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
        if kwargs.get("reduced", False):  # (1-f1-f2)*exp(-D3*b)
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
        error_msg = f"Invalid model for fitting: {model}"
        logger.error(error_msg)
        raise ValueError(error_msg)


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
        fit_result = (np.zeros(x0.shape), (0, 0))
    if timer:
        elapsed_time = time.time() - start_time
        logger.info(f"Fitting time for idx {idx}: {elapsed_time:.4f}s")
    return idx, fit_result[0], fit_result[1]


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
        logger.warning("Fit error for idx {idx}. Using zeros as result.")
        fit_result = (np.zeros(x0.shape), (0, 0))
    if timer:
        elapsed_time = time.time() - start_time
        logger.info(f"Fitting time for idx {idx}: {elapsed_time:.4f}s")
    return idx, fit_result[0], fit_result[1]


class MonoExpFitModel(AbstractFitModel):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.reduced = kwargs.get("reduced", False)
        self.fit_t1 = False
        self.mixing_time = kwargs.get("mixing_time", None)
        if self.mixing_time:
            self.fit_t1 = kwargs.get("fit_t1", True)

    @property
    def args(self) -> list:
        _args = []
        if not self.reduced:
            _args.append("S0")
        _args.append("D1")
        if self.fit_t1:
            _args.append("T1")
        return _args

    def model(self, b_values: np.ndarray, *args, **kwargs):
        """Mono-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f/S0 , D, (mixing_time)) or
                (D, (mixing_time) for reduced. See self.args for
                necessary arguments.
        """
        f = 0
        if self.reduced:
            f += np.exp(-np.kron(b_values, abs(args[0])))
        else:
            f += args[0] * np.exp(-np.kron(b_values, abs(args[1])))
        # add t1 fitting term
        if self.mixing_time:
            f *= np.exp(-args[-1] / self.mixing_time)
        return f

    def fit(
        self, idx: int | tuple, signal: np.ndarray, b_values: np.ndarray, **kwargs
    ) -> tuple:
        """Fit the exponential model to the signal.

        Args:
            idx (int): Index of the voxel.
            signal (np.ndarray): Signal data.
            x0 (np.ndarray): Initial guess for the fit.
            lb (np.ndarray): Lower bounds for the fit.
            ub (np.ndarray): Upper bounds for the fit.
            b_values (np.ndarray): B-values of the signal.
            max_iter (int): Maximum number of iterations.
            **kwargs: Additional optional keyword arguments.

        Returns:
            tuple: (idx, fit_result, fit_covariance)
        """
        x0 = kwargs.get("x0", np.array([]))
        if x0.size == 0:
            error_msg = "No starting value provided"
            logger.error(error_msg)
            raise ValueError

        timer = kwargs.get("timer", False)
        if timer:
            start_time = time.time()

        try:
            fit_result = curve_fit(
                self.model,
                b_values,
                signal,
                p0=kwargs.get("x0"),
                bounds=(kwargs.get("lb"), kwargs.get("ub")),
                max_nfev=kwargs.get("max_iter"),
                method=kwargs.get("algorithm", "trf"),
            )
        except (RuntimeError, ValueError):
            fit_result = (np.zeros(x0.shape), (0, 0))

        if timer:
            elapsed_time = time.time() - start_time
            logger.info(f"Fitting time for idx {idx}: {elapsed_time:.4f}s")

        return idx, fit_result[0], fit_result[1]


class BiExpFitModel(MonoExpFitModel):
    """

    TODO: add fix_t1
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

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.fix_d: bool = kwargs.get("fix_d", False)

    @property
    def args(self) -> list:
        _args = [
            "f1",
            "D1",
        ]
        if not self.reduced:
            _args.extend(["f2", "D2"])
        else:
            _args.append("D2")
        if self.fit_t1:
            _args.append("T1")
        return _args

    def model(self, b_values, *args, **kwargs):
        """Bi-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f1, D1, f2, D2, (mixing_time)) or
                (f1, D1, D2, (mixing_time)) for reduced.
        """
        # Add fist component f1*exp(-D1*b)
        f = args[0] * np.exp(-np.kron(b_values, abs(args[1])))
        # Add second component f
        if self.reduced:  # (1-f1)*exp(-D2*b)
            if self.fix_d:
                f += (1 - args[0]) * np.exp(
                    -np.kron(b_values, abs(kwargs.get("fixed_d", 0)))
                )
            else:
                f += (1 - args[0]) * np.exp(-np.kron(b_values, abs(args[2])))
        else:  # f2*exp(-D2*b)
            if self.fix_d:
                f += args[2] * np.exp(-np.kron(b_values, abs(kwargs.get("fixed_d", 0))))
            else:
                f += args[2] * np.exp(-np.kron(b_values, abs(args[3])))
        # Add t1 fitting term
        if self.fit_t1:  # * exp(-t1/mixing_time)
            f *= np.exp(-args[-1] / self.mixing_time)
        return f

    def fit(
        self, idx: int | tuple, signal: np.ndarray, b_values: np.ndarray, **kwargs
    ) -> tuple:
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
        if not self.fix_d:
            return super().fit(idx, signal, b_values, **kwargs)
        else:
            fixed_d: np.ndarray = kwargs.get("fixed_d")
            x0 = kwargs.get("x0", np.array([]))
            if x0.size == 0:
                error_msg = "No starting value provided"
                logger.error(error_msg)
                raise ValueError

            timer = kwargs.get("timer", False)
            if timer:
                start_time = time.time()

            model = partial(self.model, fixed_d=fixed_d)

            try:
                fit_result = curve_fit(
                    model,
                    b_values,
                    signal,
                    p0=kwargs.get("x0"),
                    bounds=(kwargs.get("lb"), kwargs.get("ub")),
                    max_nfev=kwargs.get("max_iter"),
                    method=kwargs.get("algorithm", "trf"),
                )
            except (RuntimeError, ValueError):
                fit_result = (np.zeros(x0.shape), (0, 0))

            if timer:
                elapsed_time = time.time() - start_time
                logger.info(f"Fitting time for idx {idx}: {elapsed_time:.4f}s")

            return idx, fit_result[0], fit_result[1]


class TriExpFitModel(BiExpFitModel):
    """

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

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    @property
    def args(self) -> list:
        _args = ["f1", "D1", "f2", "D2"]
        if not self.reduced:
            _args.append("f3")
        _args.append("D3")
        if self.fit_t1:
            _args.append("T1")
        return _args

    def model(self, b_values, *args, **kwargs):
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
        if self.reduced:  # (1-f1-f2)*exp(-D3*b)
            if self.fix_d:
                f += (1 - args[0] - args[2]) * np.exp(
                    -np.kron(b_values, abs(kwargs.get("fixed_d", 0)))
                )
            else:
                f += (1 - args[0] - args[2]) * np.exp(-np.kron(b_values, abs(args[4])))
        else:
            if self.fix_d:
                f += args[4] * np.exp(-np.kron(b_values, abs(kwargs.get("fixed_d", 0))))
            else:
                f += args[4] * np.exp(-np.kron(b_values, abs(args[5])))
        # Add t1 fitting term
        if self.fit_t1:  # * exp(-t1/mixing_time)
            f *= np.exp(-args[-1] / self.mixing_time)
        return f

    def fit(
        self, idx: int | tuple, signal: np.ndarray, b_values: np.ndarray, **kwargs
    ) -> tuple:
        return super().fit(idx, signal, b_values, **kwargs)


def get_model_class(model_name: str):
    """Get the model class by name.

    Args:
        model_name (str): Name of the model.

    Returns:
        FitModel: Model class.
    """
    model_classes = {
        "mono": MonoExpFitModel,
        "bi": BiexpFitModel,
        "tri": TriexpFitModel,
    }
    if model_name not in model_classes:
        error_msg = f"{model_name} is not a valid model."
        logger.error(error_msg)
        raise ValueError(error_msg)
    return model_classes[model_name]
