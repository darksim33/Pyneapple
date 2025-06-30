from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.optimize import curve_fit
from ..utils.logger import logger


class AbstractFitModel(ABC):
    def __init__(self, name:str, **kwargs):
        self._name = ""
        self._args = None

    @property
    @abstractmethod
    def name(self):
        return self._name

    @name.setter
    def name(self, name:str):
        self._name = name

    @property
    @abstractmethod
    def args(self) -> None | list:
        return self._args

    @abstractmethod
    def model(self, b_values: np.ndarray, *args):
        pass

    @abstractmethod
    def fit(self, idx: int | tuple, signal: np.ndarray, b_values: np.ndarray, **kwargs) -> tuple:
        pass

class MonoFitModel(AbstractFitModel):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.reduced = kwargs.get("reduced", False)
        self.fit_t1 = False
        self.mixing_time = kwargs.get("mixing_time", None)
        if self.mixing_time:
            self.fit_t1 = kwargs.get("fit_t1", True)
        self.fixed_d = kwargs.get("fixed_d", None)

    @property
    def args(self) -> list:
        _args = []
        if not self.reduced:
            _args.append("D1")
        _args.append("S0")
        if self.fit_t1:
            _args.append("T1")
        return _args


    def model(self, b_values: np.ndarray, *args):
        """Mono-exponential model function.

        Args:
            b_values (np.ndarray): B-values.
            *args: Arguments of shape (f/S0 , D, (mixing_time)) or
                (D, (mixing_time) for reduced.
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

    @staticmethod
    def fit(self, idx: int | tuple, signal: np.ndarray, b_values: np.ndarray, **kwargs) -> tuple:
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
        x0=kwargs.get("x0", np.array([]))
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
            fit_result = (np.zeros(x0), (0, 0))

        if timer:
            elapsed_time = time.time() - start_time
            logger.info(f"Fitting time for idx {idx}: {elapsed_time:.4f}s")

        return idx, fit_result[0], fit_result[1]
