from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from functools import partial
from typing import Callable

# from ..models import IVIM
from .ivim import IVIMParams

# CURRENTLY NOT WORKING


class IDEALParams(IVIMParams):
    """IDEAL fitting Parameter class.

    Attributes:
        tolerance (np.ndarray): Tolerance for IDEAL step boundaries in relative values.
        dimension_steps (np.ndarray): Steps for dimension reduction in IDEAL.
        segmentation_threshold (float): Threshold for segmentation in IDEAL.

    Methods:

    """

    def __init__(
        self,
        params_json: Path | str = None,
    ):
        """
        IDEAL fitting Parameter class.

        Args:
            params_json (Path, str): Parameter json file containing basic fitting
            parameters.

        """
        self.tolerance = None
        self.dimension_steps = None
        self.segmentation_threshold = None
        super().__init__(params_json)
        self.fit_function = IVIM.fit
        self.fit_model = IVIM.wrapper

    @property
    def fit_function(self):
        return partial(
            self._fit_function,
            b_values=self.get_basis(),
            n_components=self.n_components,
            max_iter=self.max_iter,
            TM=None,
            scale_image=self.scale_image if isinstance(self.scale_image, str) else None,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        self._fit_function = method

    @property
    def fit_model(self):
        return self._fit_model(
            n_components=self.n_components,
            TM=self.mixing_time,
            scale_image=self.scale_image if isinstance(self.scale_image, str) else None,
        )

    @fit_model.setter
    def fit_model(self, method: Callable):
        self._fit_model = method

    @property
    def dimension_steps(self):
        return self._dimension_steps

    @dimension_steps.setter
    def dimension_steps(self, value):
        if isinstance(value, list):
            steps = np.array(value)
        elif isinstance(value, np.ndarray):
            steps = value
        elif value is None:
            # TODO: Special None Type handling? (IDEAL)
            steps = None
        else:
            raise TypeError()
        # Sort Dimension steps
        self._dimension_steps = (
            np.array(sorted(steps, key=lambda x: x[1], reverse=True))
            if steps is not None
            else None
        )

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: list | np.ndarray | None):
        """
        Tolerance for IDEAL step boundaries in relative values.

        value: list | np.ndarray
            All stored values need to be floats with 0 < value < 1
        """
        if isinstance(value, list):
            self._tolerance = np.array(value)
        elif isinstance(value, np.ndarray):
            self._tolerance = value
        elif value is None:
            self._tolerance = None
        else:
            raise TypeError()

    @property
    def segmentation_threshold(self):
        return self._segment_threshold

    @segmentation_threshold.setter
    def segmentation_threshold(self, value: float | None):
        if isinstance(value, float):
            self._segment_threshold = value
        elif value is None:
            self._segment_threshold = 0.025
        else:
            raise TypeError()

    def get_basis(self):
        return np.squeeze(self.b_values)

    def get_pixel_args(self, img: np.ndarray, seg: np.ndarray, *args) -> zip:
        # Behaves the same way as the original parent funktion with the difference that instead of Nii objects
        # np.ndarrays are passed. Also needs to pack all additional fitting parameters [x0, lb, ub]
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

    def interpolate_start_values_2d(
        self, boundary: np.ndarray, matrix_shape: np.ndarray, n_pools: int | None = None
    ) -> np.ndarray:
        """
        Interpolate starting values for the given boundary.

        boundary: np.ndarray of shape(x, y, z, n_variables).
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        # if boundary.shape[0:1] < matrix_shape:
        boundary_new = np.zeros(
            (matrix_shape[0], matrix_shape[1], boundary.shape[2], boundary.shape[3])
        )
        arg_list = zip(
            ([i, j] for i, j in zip(*np.nonzero(np.ones(boundary.shape[2:4])))),
            (
                boundary[:, :, i, j]
                for i, j in zip(*np.nonzero(np.ones(boundary.shape[2:4])))
            ),
        )
        func = partial(self.interpolate_array_multithreading, matrix_shape=matrix_shape)
        results = multithreader(func, arg_list, n_pools=n_pools)
        return sort_interpolated_array(results, array=boundary_new)

    def interpolate_img(
        self,
        img: np.ndarray,
        matrix_shape: np.ndarray | list | tuple,
        n_pools: int | None = None,
    ) -> np.ndarray:
        """
        Interpolate image to desired size in 2D.

        img: np.ndarray of shape(x, y, z, n_bvalues)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        # get new empty image
        img_new = np.zeros(
            (matrix_shape[0], matrix_shape[1], img.shape[2], img.shape[3])
        )
        # get x*y image planes for all slices and decay points
        arg_list = zip(
            ([i, j] for i, j in zip(*np.nonzero(np.ones(img.shape[2:4])))),
            (img[:, :, i, j] for i, j in zip(*np.nonzero(np.ones(img.shape[2:4])))),
        )
        func = partial(
            self.interpolate_array_multithreading,
            matrix_shape=matrix_shape,
        )
        results = multithreader(func, arg_list, n_pools=n_pools)
        return sort_interpolated_array(results, array=img_new)

    def interpolate_seg(
        self,
        seg: np.ndarray,
        matrix_shape: np.ndarray | list | tuple,
        threshold: float,
        n_pools: int | None = 4,
    ) -> np.ndarray:
        """
        Interpolate segmentation to desired size in 2D and apply threshold.

        seg: np.ndarray of shape(x, y, z)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        seg_new = np.zeros((matrix_shape[0], matrix_shape[1], seg.shape[2], 1))

        # get x*y image planes for all slices and decay points
        arg_list = zip(
            ([i, j] for i, j in zip(*np.nonzero(np.ones(seg.shape[2:4])))),
            (seg[:, :, i, j] for i, j in zip(*np.nonzero(np.ones(seg.shape[2:4])))),
        )
        func = partial(
            self.interpolate_array_multithreading,
            matrix_shape=matrix_shape,
        )
        results = multithreader(func, arg_list, n_pools=n_pools)
        seg_new = sort_interpolated_array(results, seg_new)

        # Make sure Segmentation is binary
        seg_new[seg_new < threshold] = 0
        seg_new[seg_new > threshold] = 1

        # Check seg size. Needs to be M x N x Z x 1
        while len(seg_new.shape) < 4:
            seg_new = np.expand_dims(seg_new, axis=len(seg_new.shape))
        return seg_new

    @staticmethod
    def interpolate_array_multithreading(
        idx: tuple | list, array: np.ndarray, matrix_shape: np.ndarray
    ):
        # Cv-less version of interpolate_image
        # def interpolate_array(arr: np.ndarray, shape: np.ndarray):
        #     """Interpolate 2D array to new shape."""
        #
        #     x, y = np.meshgrid(
        #         np.linspace(0, 1, arr.shape[1]), np.linspace(0, 1, arr.shape[0])
        #     )
        #     x_new, y_new = np.meshgrid(
        #         np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0])
        #     )
        #     points = np.column_stack((x.flatten(), y.flatten()))
        #     values = arr.flatten()
        #     new_values = griddata(points, values, (x_new, y_new), method="cubic")
        #     return np.reshape(new_values, shape)

        def interpolate_array_cv(arr: np.ndarray, shape: np.ndarray):
            return cv2.resize(arr, shape, interpolation=cv2.INTER_CUBIC)

        # def interpolate_array_scipy

        array = interpolate_array_cv(array, matrix_shape)
        return idx, array

    def eval_fitting_results(self, results: np.ndarray, **kwargs) -> dict:
        """
        Evaluate fitting results for the IDEAL method.

        Parameters
        ----------
            results
                Pass the results of the fitting process to this function
            seg: NiiSeg
                Get the shape of the spectrum array
        """
        # TODO Needs rework

        coordinates = kwargs.get("seg").get_seg_indices("nonzero")
        # results_zip = list(zip(coordinates, results[coordinates]))
        results_zip = zip(
            (coord for coord in coordinates), (results[coord] for coord in coordinates)
        )
        fit_results = super().eval_fitting_results(results_zip)
        return fit_results
