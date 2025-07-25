"""Multi-Exponential Models for fitting on the CPU.

This module provides explicit multi-exponential model classes for fitting on the CPU, including mono-, bi-, and
tri-exponential models. These models are implemented as classes for compatibility with multiprocessing
(avoiding pickling errors from dynamic method parsing). For segmented fitting, the last component is always used as a
fixed component, and bounds must be set accordingly.

Classes:
    MonoExpFitModel: Mono-exponential model class.
    BiExpFitModel: Bi-exponential model class.
    TriExpFitModel: Tri-exponential model class.

Functions:
    get_model_class(model_name): Returns the model class by name ('mono', 'bi', 'tri').

Notes:
    - Multiprocessing does not support dynamic parsing of methods. Therefore, models are implemented as explicit classes
    and selected by string comparison.
    - The IVIM and IVIMFixedComponent classes have been removed in favor of the new wrapper model classes.

Version History:
    1.5.0 (2024-12-06): Created the wrapper model classes. Added MonoExpFitModel, BiExpFitModel, TriExpFitModel, and
    get_model_class function. Removed IVIM and IVIMFixedComponent classes.
    1.6.1 (2025-07-08): Reworked model wrappers to classes and added S0 fitting option.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import curve_fit
from functools import partial

from ..utils.logger import logger
from .model import AbstractFitModel


class MonoExpFitModel(AbstractFitModel):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.fit_reduced = kwargs.get("fit_reduced", False)
        self.fit_t1 = False
        self.mixing_time = kwargs.get("mixing_time", None)
        if self.mixing_time:
            self.fit_t1 = kwargs.get("fit_t1", True)

    @property
    def args(self) -> list:
        _args = []
        if not self.fit_reduced:
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
                (D, (mixing_time) for fit_reduced. See self.args for
                necessary arguments.
        """
        f = 0
        if self.fit_reduced:
            f += np.exp(-np.kron(b_values, abs(args[0])))
        else:
            f += args[0] * np.exp(-np.kron(b_values, abs(args[1])))

        # Add t1 fitting term
        f = self.add_t1(f, *args, **kwargs)

        return f

    def add_t1(self, f, *args, **kwargs):
        """Add T1 term or fixed T1 term to the model."""
        if self.fit_t1 and not abs(
                kwargs.get("fixed_t1", False)
        ):  # * exp(-t1/mixing_time)
            f *= np.exp(-args[-1] / self.mixing_time)
        elif self.fit_t1 and abs(kwargs.get("fixed_t1", False)):
            f *= np.exp(-kwargs.get("fixed_t1"))
        return f

    def fit(self, idx: int | tuple, signal: np.ndarray, *args, **kwargs) -> tuple:
        """Fit the exponential model to the signal.

        Args (non-optional):
            idx (int): Index of the voxel.
            signal (np.ndarray): Signal data (y-data).
            b_values: np.ndarray: B-values of the signal (x-data).
            x0 (np.ndarray): Initial guess for the fit.
            lb (np.ndarray): Lower bounds for the fit.
            ub (np.ndarray): Upper bounds for the fit.
            b_values (np.ndarray): B-values of the signal.
            max_iter (int): Maximum number of iterations.
        Args (optional):
            *args: Additional positional arguments.
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
                kwargs.get("b_values"),
                signal,
                p0=x0,
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
    """Bi-exponential model for fitting.

    Args:
        **kwargs: Additional keyword arguments.
            "fit_reduced" (bool): Reduced model with only one component.
            "mixing_time" (float, None): Mixing time value. Needed for T1 fitting.
            "fixed_d" (float, None): Fixed D value for the second component.
            "fit_S0" (bool): Fit S0 value instead of to f instead.
    Models:
        f       = f1 * exp(-D1 * b) + f2 * exp(-D2 * b)
        f_t1    = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) * exp(-t1 / mixing_time)
            with: args[0] = f1, args[1] = D1, args[2] = f2, args[3] = D2,
            (args[4] = mixing_time)
        f_red   = f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b)
        f_red_t1= f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b) * exp(-t1 / mixing_time)
            with: args[0] = f1, args[1] = D1, args[2] = D2, (args[3] = mixing_time)
        f_S0 = (f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b)) * S0
        f_S0_t1 = (f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b)) * S0 * exp(-t1 / mixing_time)
            with: args[0] = f1, args[1] = D1, args[2] = D2, args[3] = S0 (args[4] = mixing_time)
    """

    def __init__(self, name: str, **kwargs):
        self.fit_S0 = False
        super().__init__(name, **kwargs)
        self.fix_d: bool = kwargs.get("fix_d", False)
        if self.fit_reduced and kwargs.get("fit_S0", False):
            error_msg = ("You cannot fit S0 in fit_reduced model.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif not self.fit_reduced and kwargs.get("fit_S0", False):
            self.fit_S0 = True

    @property
    def args(self) -> list:
        _args = [
            "f1",
            "D1",
        ]
        if not self.fit_reduced and not self.fit_S0:
            _args.extend(["f2", "D2"])
        else:
            _args.append("D2")
        if self.fit_S0:
            _args.append("S0")
        if self.fit_t1:
            _args.append("T1")
        return _args

    def model(self, b_values, *args, **kwargs):
        """Bi-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f1, D1, f2, D2, (mixing_time)) or
                (f1, D1, D2, (mixing_time)) for fit_reduced.
        """
        # Add fist component f1*exp(-D1*b)
        f = args[0] * np.exp(-np.kron(b_values, abs(args[1])))
        # Add second component f
        if self.fit_reduced or self.fit_S0:  # (1-f1)*exp(-D2*b)
            if self.fix_d:
                f += (1 - args[0]) * np.exp(
                    -np.kron(b_values, abs(kwargs.get("fixed_d", 0)))
                )
            else:
                f += (1 - args[0]) * np.exp(-np.kron(b_values, abs(args[2])))
            if self.fit_S0:
                f *= args[3]
        else:  # f2*exp(-D2*b)
            if self.fix_d:
                f += args[2] * np.exp(-np.kron(b_values, abs(kwargs.get("fixed_d", 0))))
            else:
                f += args[2] * np.exp(-np.kron(b_values, abs(args[3])))

        # Add t1 fitting term
        f = self.add_t1(f, *args, **kwargs)

        return f

    def fit(self, idx: int | tuple, signal: np.ndarray, *args, **kwargs) -> tuple:
        """Standard exponential model fit using "curve_fit" with additional options for
        fixed parameters.

        Args:
            idx (int): Index of the voxel.
            signal (np.ndarray): Signal data (y-data).
            fixed_values/*args (list): List of fixed values for the fit.
                (fixed_d, fixed_t1)
            b_values: np.ndarray: B-values of the signal (x-data).
            x0 (np.ndarray): Initial guess for the fit.
            lb (np.ndarray): Lower bounds for the fit.
            ub (np.ndarray): Upper bounds for the fit.
            model (str): Model name.
            b_values (np.ndarray): B-values of the signal.
            max_iter (int): Maximum number of iterations.
            timer (bool): Timer for the fit.
            **kwargs: Additional keyword arguments.
                fit_reduced (bool): Reduced model for S/S0 fitting replacing one fraction
                    (sum(f)=1).
                mixing_time (float): Mixing time value. Needed for T1 fitting.
        Returns:
            idx (int): Index of the voxel.
            fit_result (np.ndarray): Fit result holding only estimated parameters.
        """
        if not self.fix_d:
            return super().fit(idx, signal, *args, **kwargs)
        else:
            # Get fixed values from kwargs and parse them as kwargs to the model
            fixed_values = kwargs.get("fixed_values")
            fixed_d: np.ndarray = fixed_values[0]
            model = partial(self.model, fixed_d=fixed_d)
            if len(fixed_values) > 1:
                fixed_t1: np.ndarray = fixed_values[1]
                model = partial(model, fixed_t1=fixed_t1)

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
                    model,
                    kwargs.get("b_values"),
                    signal,
                    p0=x0,
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
    """Tri-exponential model for fitting.

    Args:
        **kwargs: Additional keyword arguments.
            "fit_reduced" (bool): Reduced model with only one component.
            "mixing_time" (float, None): Mixing time value. Needed for T1 fitting.
            "fixed_d" (float, None): Fixed D value for the second component.
            "fit_S0" (bool): Fit S0 value instead of to f instead.

    Models:
        f       = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + f3 * exp(-D3 * b)
        f_t1    = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + f3 * exp(-D3 * b)
            * exp(-t1 / mixing_time) with: args[0] = f1, args[1] = D1, args[2] = f2,
            args[3] = D2, args[4] = f3, args[5] = D3, (args[6] = mixing_time)
        f_red   = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b)
        f_red_t1= f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b)
            * exp(-t1 / mixing_time) with: args[0] = f1, args[1] = D1, args[2] = f2,
            args[3] = D2, args[4] = D3, (args[5] = mixing_time)
        f_S0 = (f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b)) * S0
        f_S0_t1 = (f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b)) * S0 *
            exp(-t1 / mixing_time) with: args[0] = f1, args[1] = D1, args[2] = f2,
            args[3] = D2, args[4] = D3, args[5] = S0, (args[6] = mixing_time)
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    @property
    def args(self) -> list:
        _args = ["f1", "D1", "f2", "D2"]
        if not self.fit_reduced and not self.fit_S0:
            _args.append("f3")
        _args.append("D3")
        if self.fit_S0:
            _args.append("S0")
        if self.fit_t1:
            _args.append("T1")
        return _args

    def model(self, b_values, *args, **kwargs):
        """Tri-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f1 , D1, f2, D2, f3, D3, (mixing_time)) or
                    (f1, D1, f2, D2, D3, (mixing_time)) for fit_reduced.
        """
        # Add first and second component f1*exp(-D1*b) + f2*exp(-D2*b)
        f = args[0] * np.exp(-np.kron(b_values, abs(args[1]))) + args[2] * np.exp(
            -np.kron(b_values, abs(args[3]))
        )
        if self.fit_reduced or self.fit_S0:  # (1-f1-f2)*exp(-D3*b)
            if self.fix_d:
                f += (1 - args[0] - args[2]) * np.exp(
                    -np.kron(b_values, abs(kwargs.get("fixed_d", 0)))
                )
            else:
                f += (1 - args[0] - args[2]) * np.exp(-np.kron(b_values, abs(args[4])))
            if self.fit_S0:
                f *= args[5]
        else:
            if self.fix_d:
                f += args[4] * np.exp(-np.kron(b_values, abs(kwargs.get("fixed_d", 0))))
            else:
                f += args[4] * np.exp(-np.kron(b_values, abs(args[5])))

        # Add t1 fitting term
        f = self.add_t1(f, *args, **kwargs)
        return f

    def fit(self, idx: int | tuple, signal: np.ndarray, *args, **kwargs) -> tuple:
        return super().fit(idx, signal, *args, **kwargs)


def get_model_class(model_name: str):
    """Get the model class by name.

    Args:
        model_name (str): Name of the model.

    Returns:
        FitModel: Model class.
    """
    model_classes = {
        "mono": MonoExpFitModel,
        "bi": BiExpFitModel,
        "tri": TriExpFitModel,
    }
    if model_name not in model_classes:
        error_msg = f"{model_name} is not a valid model."
        logger.error(error_msg)
        raise ValueError(error_msg)
    return model_classes[model_name]
