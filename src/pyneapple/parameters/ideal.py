from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from .ivim import IVIMParams
from ..utils.logger import logger
from radimgarray import RadImgArray


class IDEALParams(IVIMParams):
    def __init__(self, file: Path | str | None = None):
        self.ideal_dims = 2
        self.dim_steps = np.array([], dtype=np.int32)
        self.step_tol: list[float] | None = None
        self.seg_threshold: float | None = None  # Segmentation threshold
        self.interpolation: int | None = cv2.INTER_CUBIC
        super().__init__(file)

    @property
    def dim_steps(self) -> np.ndarray:
        return self._dimension_steps

    @dim_steps.setter
    def dim_steps(self, value: list[int] | np.ndarray) -> None:
        if isinstance(value, list):
            _dimension_steps = np.array(value, dtype=np.int32)
        elif isinstance(value, np.ndarray):
            _dimension_steps = value.astype(np.int32)
        elif value is None:
            self._dimension_steps = np.array([], dtype=np.int32)
            return
        else:
            error_msg = (
                f"Expected list or numpy array for dimension_steps, got {type(value)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._dimension_steps = np.array(sorted(_dimension_steps, key=lambda x: x[1]))

    @property
    def step_tol(self) -> np.ndarray | None:
        return self._step_tolerance

    @step_tol.setter
    def step_tol(self, value: list[float] | np.ndarray | float) -> np.ndarray | None:
        if isinstance(value, list):
            self._step_tolerance = np.array(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            self._step_tolerance = value.astype(np.float32)
        elif isinstance(value, float):
            self._step_tolerance = np.array([value], dtype=np.float32)
        elif value is None:
            self._step_tolerance = None
            return
        else:
            error_msg = (
                f"Expected list or numpy array for step_tolerance, got {type(value)}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if self.fit_model.args:
            if len(self._step_tolerance) != len(self.fit_model.args):
                warn_msg = (
                    f"Length of step_tolerance ({len(self._step_tolerance)}) "
                    f"does not match number of fitted arguments "
                    f"({len(self.fit_model.args)})."
                )
                logger.warning(warn_msg)

    @property
    def seg_threshold(self):
        return self._segment_threshold

    @seg_threshold.setter
    def seg_threshold(self, value: float | None):
        if isinstance(value, (float, np.floating)):
            self._segment_threshold = value
        elif value is None:
            self._segment_threshold = 0.025
        else:
            error_msg = f"Expected float for segmentation_threshold, got {type(value)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

    def get_pixel_args(self, img: np.ndarray, seg: np.ndarray, *args) -> zip:
        """Zip pixel arguments for fitting.

        Behaves the same way as the original parent funktion with the difference that
        also additional fitting parameters [x0, lb, ub] are zipped.

        Args:
            img (np.ndarray): Image data (4D)
            seg (np.ndarray): Segmentation data (4D [x,y,z,1])
            *args: Additional arguments
        Returns:
            zip: Zip of tuples containing pixel arguments [(i, j, k), img[i, j, k, :]]
        """
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
            (img[i, j, k, :] for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
            (
                args[0][i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
            (
                args[1][i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
            (
                args[2][i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
        )
        return pixel_args

    def interpolate_array(
        self, array: np.ndarray, step_idx: int, **kwargs
    ) -> np.ndarray:
        """Interpolate array with min 3 dimensions using CV.

        Args:
            array (np.ndarray): Base array to interpolate from.
            step_idx (int): Number of dimension step to interpolate.
            **kwargs: Additional keyword arguments for interpolation.
                matrix_shape (tuple): Shape of the matrix to interpolate.
                interpolation (int): Interpolation method to use (default: cv2.INTER_CUBIC).
        """
        matrix_shape = kwargs.get("matrix_shape", self.dim_steps[step_idx])
        if self.ideal_dims == 2:
            _array = array.reshape(array.shape[0], array.shape[1], -1)

            interpolated_array = cv2.resize(
                _array,
                matrix_shape,
                interpolation=self.interpolation,
            )
            return interpolated_array.reshape(
                (*matrix_shape, array.shape[2], array.shape[3]),
            )

        else:
            error_msg = "Currently only 2D interpolation is supported for IDEALParams."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)

    def interpolate_img(
        self, img: np.ndarray | RadImgArray, step_idx: int, **kwargs
    ) -> RadImgArray:
        """Interpolate image data.

        Args:
            img (np.ndarray | RadImgArray): Image data to interpolate.
            step_idx (int): Index of the step to interpolate image for.

        Returns:
            RadImgArray: Interpolated image data.
        """

        _img = self.interpolate_array(img, step_idx, **kwargs)
        if isinstance(img, RadImgArray):
            if isinstance(_img, np.ndarray):
                _img = RadImgArray(_img, img.info)
        else:
            _img = RadImgArray(_img)

        return _img

    def interpolate_seg(
        self, seg: np.ndarray | RadImgArray, step_idx: int, **kwargs
    ) -> RadImgArray:
        """Interpolate segmentation data.

        Args:
            seg (np.ndarray | RadImgArray): Image data to interpolate.
            step_idx (int): Index of the step to interpolate segmentation for.

        Returns:
            RadImgArray: Interpolated segmentation data.
        """
        _seg = self.interpolate_img(seg, step_idx, **kwargs)
        # Make sure Segmentation is binary
        _seg[_seg < self.seg_threshold] = 0
        _seg[_seg > self.seg_threshold] = 1
        return _seg

    def get_boundaries(
        self, step_idx: int, result: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate boundaries for the given step index.

        Args:
            step_idx (int): Index of the step to interpolate boundaries for.
            result (np.ndarray): Results from the previous fitting step.

        Returns:
            tuple: Interpolated start values, lower bounds, and upper bounds.
        """
        x0 = self.boundaries.start_values(self.fit_model.args)
        lb = self.boundaries.lower_bounds(self.fit_model.args)
        ub = self.boundaries.upper_bounds(self.fit_model.args)

        if step_idx > 0:
            x0 = self.interpolate_img(result, step_idx)
            ub = x0 * (1 + self.step_tol)
            lb = x0 * (1 - self.step_tol)

        return x0, lb, ub

    def sort_fit_results(self, img: RadImgArray, results: list) -> np.ndarray:
        """Sort fit results based on the dimension steps.

        Args:
            img (RadImgArray): Image data used for sorting.
            results (list): List of fit results to be sorted.

        Returns:
            np.ndarray: Sorted fit results.
        """
        _array = np.zeros(
            (img.shape[0], img.shape[1], img.shape[2], len(self.fit_model.args)),
            dtype=np.float32,
        )
        for element in results:
            _array[element[0]] = element[1]
        return _array
