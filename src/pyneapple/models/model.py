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
    def model(self, b_values: np.ndarray, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, idx: int | tuple, signal: np.ndarray, b_values: np.ndarray, **kwargs) -> tuple:
        pass

