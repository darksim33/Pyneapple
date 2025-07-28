from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.optimize import curve_fit
from ..utils.logger import logger


class AbstractFitModel(ABC):
    def __init__(self, name: str = "", **kwargs):
        self._name = name
        self._args = None

    @property
    def name(self):
        """Get or set the name of the model."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    @abstractmethod
    def args(self) -> None | list:
        """Get the arguments used in the current configured model."""
        return self._args

    @abstractmethod
    def model(self, b_values: np.ndarray, *args, **kwargs):
        """Return the model function for the given b-values."""
        pass

    @abstractmethod
    def fit(self, idx: int | tuple, signal: np.ndarray, *args, **kwargs) -> tuple:
        """Fit the model to the signal data and return the fitted parameters.

        Args:
            idx (int | tuple): Index of the voxel to be fitted
            signal (np.ndarray): Signal decay to be fitted
            *args: Additional arguments for the fitting function
            **kwargs: Keyword arguments for the fitting function
                b_values (np.ndarray): B-values for the fitting (!not optional!)
        """
        pass


class BaseFitModel(AbstractFitModel):
    """Base class for all models."""

    def __init__(self, name: str = "", **kwargs):
        super().__init__(name, **kwargs)
        self._args = None

    @property
    def args(self) -> None | list:
        """Get the arguments used in the current configured model."""
        return self._args

    def model(self, b_values: np.ndarray, *args, **kwargs):
        """Return the model function for the given b-values."""
        pass

    def fit(self, idx: int | tuple, signal: np.ndarray, *args, **kwargs) -> tuple:
        """Fit the model to the signal data and return the fitted parameters."""
        pass
