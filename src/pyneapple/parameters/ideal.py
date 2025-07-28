from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from functools import partial
from typing import Callable

from .ivim import IVIMParams
from ..utils.logger import logger


class IDEALParams(IVIMParams):
    def __init__(self, file: Path | str | None = None, *args, **kwargs):
        self.ideal_dims = 2
        self.step_tolerance: list[float] | None = None
        self.dimension_steps = None
        self.segmentation_threshold: float | None = None
        super().__init__(file, *args, **kwargs)

    @property
    def dimension_steps(self) -> np.ndarray | None:
        return self._dimension_steps

    @dimension_steps.setter
    def dimension_steps(self, value: list[int] | np.ndarray) -> None:
        if isinstance(value, list):
            _dimension_steps = np.array(value, dtype=np.int32)
        elif isinstance(value, np.ndarray):
            _dimension_steps = value.astype(np.int32)
        elif value is None:
            self._dimension_steps = None
            return
        else:
            error_msg = (
                f"Expected list or numpy array for dimension_steps, got {type(value)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._dimension_steps = np.array(sorted(_dimension_steps, key=lambda x: x[1]))

    @property
    def step_tolerance(self) -> np.ndarray | None:
        return self._step_tolerance

    @step_tolerance.setter
    def step_tolerance(self, value: list[float] | np.ndarray | float) -> None:
        if isinstance(value, list):
            self._step_tolerance = np.array(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            self._step_tolerance = value.astype(np.float32)
        elif isinstance(value, float):
            self._step_tolerance = np.array([value], dtype=np.float32)
        elif value is None:
            self._step_tolerance = None
        else:
            error_msg = (
                f"Expected list or numpy array for step_tolerance, got {type(value)}"
            )
            logger.warning(error_msg)
            raise TypeError(error_msg)

    @property
    def segmentation_threshold(self):
        return self._segment_threshold

    @segmentation_threshold.setter
    def segmentation_threshold(self, value: float | None):
        if isinstance(value, (float, np.floating)):
            self._segment_threshold = value
        elif value is None:
            self._segment_threshold = 0.025
        else:
            error_msg = f"Expected float for segmentation_threshold, got {type(value)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

    def interpolate_start_values(self, base_array: np.ndarray, step_idx: int, **kwargs):
        """Interpolate start values for IDEAL parameters.

        Args:
            base_array (np.ndarray): Base array to interpolate from. Prior fit results. Shape: (x,y,z,n_args).
            step_idx (int): Number of dimension steps to interpolate.
            **kwargs: Additional keyword arguments for interpolation.
                matrix_shape (tuple): Shape of the matrix to interpolate.
                interpolation (int): Interpolation method to use (default: cv2.INTER_CUBIC).
        """

        matrix_shape = kwargs.get("matrix_shape", self.dimension_steps[step_idx])

        if self.ideal_dims == 2:
            # For 2D, we use cv2.resize for interpolation
            _array = base_array.reshape(base_array.shape[0], base_array.shape[1], -1)

            resized = cv2.resize(
                _array,
                matrix_shape,
                interpolation=kwargs.get("interpolation", cv2.INTER_CUBIC),
            )
            return resized.reshape(
                (*matrix_shape, base_array.shape[2], base_array.shape[3]),
            )
        else:
            error_msg = "Currently only 2D interpolation is supported for IDEALParams."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
