import inspect

import numpy as np
from scipy import ndimage
from pathlib import Path
from functools import partial
from typing import Callable
import json
from src.exceptions import ClassMismatch
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d, griddata

from .fit import *
from .parameters import IVIMParams

# from .model import Model
from .fit import fit


class Model(object):
    """Contains fitting methods of all different models."""

    class IDEAL(object):
        @staticmethod
        def wrapper(n_components: int):
            """Creates function for IVIM model, able to fill with partial."""

            def multi_exp_model(b_values, *args):
                f = 0
                for i in range(n_components - 1):
                    f += (
                            np.exp(-np.kron(b_values, abs(args[i])))
                            * args[n_components + i]
                    )
                f += (
                        np.exp(-np.kron(b_values, abs(args[n_components - 1])))
                        # Second half containing f, except for S0 as the very last entry
                        * (1 - (np.sum(args[n_components:-1])))
                )

                return f * args[-1]  # Add S0 term for non-normalized signal

            return multi_exp_model

        @staticmethod
        def fit(
                idx: int,
                signal: np.ndarray,
                b_values: np.ndarray,
                n_components: int,
                args: np.ndarray,
                lb: np.ndarray,
                ub: np.ndarray,
                max_iter: int,
                timer: bool | None = False,
        ):
            """Standard IVIM fit using the IVIM model wrapper."""
            # start_time = time.time()

            try:
                fit_result = curve_fit(
                    Model.IDEAL.wrapper(n_components=n_components),
                    b_values,
                    signal,
                    p0=args,
                    bounds=(lb, ub),
                    max_nfev=max_iter,
                )[0]
                # if timer:
                #     print(time.time() - start_time)
            except (RuntimeError, ValueError):
                fit_result = np.zeros(args.shape)
                if timer:
                    print("Error")
            return idx, fit_result

        @staticmethod
        def printer(n_components: int, args):
            """Model printer for testing."""
            f = f""
            for i in range(n_components - 1):
                f += f"exp(-kron(b_values, abs({args[i]}))) * {args[n_components + i]} + "
            f += f"exp(-kron(b_values, abs({args[n_components - 1]}))) * (1 - (sum({args[n_components:-1]})))"
            return f"( " + f + f" ) * {args[-1]}"


class IDEALParams(IVIMParams):
    """
    IDEAL fitting Parameter class

    ...

    Attributes
    ----------
    fit_model: FitModel
        The FitModel used in IDEAL approach
        The default model is the tri-exponential Model with starting values optimized for kidney
        Parameters are as follows: D_fast, D_interm, D_slow, f_fast, f_interm, (S0)
        ! For these Models the last fraction is typically calculated from the sum of fraction
    tol: np.ndarray
        ideal adjustment tolerance for each parameter
    dimension_steps: np.ndarray
        down-sampling steps for fitting

    """

    def __init__(
        self,
        params_json: Path | str = None,
    ):
        self.tol = None
        self.dimension_steps = None
        self.segmentation_threshold = None
        super().__init__(params_json)
        self.fit_function = Model.IDEAL.fit
        self.fit_model = Model.IDEAL.wrapper

    @property
    def fit_function(self):
        return partial(
            self._fit_function,
            b_values=self.get_basis(),
            max_iter=self.max_iter,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        self._fit_function = method

    @property
    def dimension_steps(self):
        return self._dimension_steps

    @dimension_steps.setter
    def dimension_steps(self, value):
        if isinstance(value, list):
            self._dimension_steps = np.array(value)
        elif isinstance(value, np.ndarray):
            self._dimension_steps = value
        elif value is None:
            # TODO: Special None Type handling?
            self._dimension_steps = None
        else:
            raise TypeError()

    def load_from_json(self, params_json: str | Path | None = None):
        if params_json is not None:
            self.json = params_json

        with open(self.json, "r") as json_file:
            params_dict = json.load(json_file)

        # Check if .json contains Class identifier and if .json and Params set match
        if "Class" not in params_dict.keys():
            # print("Error: Missing Class identifier!")
            # return
            raise ClassMismatch("Error: Missing Class identifier!")
        elif not isinstance(self, globals()[params_dict["Class"]]):
            raise ClassMismatch("Error: Wrong parameter.json for parameter Class!")
        else:
            params_dict.pop("Class", None)
            for key, item in params_dict.items():
                # if isinstance(item, list):
                if hasattr(self, key):
                    setattr(self, key, item)
                else:
                    print(
                        f"Warning: There is no {key} in the selected Parameter set! {key} is skipped."
                    )

    def get_basis(self):
        return np.squeeze(self.b_values)

    def get_pixel_args(self, img: np.ndarray, seg: np.ndarray, *args) -> partial:
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
        self, boundary: np.ndarray, matrix_shape: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate starting values for the given boundary.

        boundary: np.ndarray of shape(x, y, z, n_variables).
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        # if boundary.shape[0:1] < matrix_shape:
        new_boundary = np.zeros(
            (matrix_shape[0], matrix_shape[1], boundary.shape[2], boundary[3])
        )
        for idx_slice, plane in enumerate(boundary.transpose(2, 0, 1, 3)):
            for idx_variable, variables in enumerate(plane.transpose(2, 0, 1)):
                new_boundary[:, :, idx_slice, idx_variable] = self.interpolate_array(
                    variables, matrix_shape
                )
        return new_boundary

    def interpolate_img(
        self, img: np.ndarray, matrix_shape: np.ndarray | list | tuple
    ) -> np.ndarray:
        """
        Interpolate image to desired size in 2D.

        img: np.ndarray of shape(x, y, z, n_bvalues)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        new_image = np.zeros(
            (matrix_shape[0], matrix_shape[1], img.shape[2], img.shape[3])
        )
        for idx_slice, plane in enumerate(img.transpose(2, 0, 1, 3)):
            plane = np.array(plane)  # needed?
            for idx_decay, decay in enumerate(plane.transpose(2, 0, 1)):
                new_image[:, :, idx_slice, idx_decay] = self.interpolate_array(
                    decay, matrix_shape
                )
        return new_image

    def interpolate_seg(
        self, seg: np.ndarray, matrix_shape: np.ndarray | list | tuple, threshold: float
    ) -> np.ndarray:
        """
        Interpolate segmentation to desired size in 2D and apply threshold.

        seg: np.ndarray of shape(x, y, z)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        seg_new = np.zeros((matrix_shape[0], matrix_shape[1], seg.shape[2]))
        for idx_slice, plane in enumerate(seg.transpose(2, 0, 1, 3)):
            seg_new[:, :, idx_slice] = self.interpolate_array(plane, matrix_shape)
        seg_new[seg_new < threshold] = 0
        # Check seg size. Needs to be M x N x Z x 1
        while len(seg_new.shape) < 4:
            seg_new = np.expand_dims(seg_new, axis=len(seg_new.shape) - 1)
        return seg_new

    @staticmethod
    def interpolate_array(array: np.ndarray, matrix_shape: np.ndarray):
        """Interpolate 2D array to new shape"""

        x, y = np.meshgrid(
            np.linspace(0, 1, array.shape[1]), np.linspace(0, 1, array.shape[0])
        )
        x_new, y_new = np.meshgrid(
            np.linspace(0, 1, matrix_shape[1]), np.linspace(0, 1, matrix_shape[0])
        )
        points = np.column_stack((x.flatten(), y.flatten()))
        values = array.flatten()
        new_values = griddata(points, values, (x_new, y_new), method="cubic")
        return np.reshape(new_values, matrix_shape)


def fit_ideal_new(
    nii_img: Nii, nii_seg: NiiSeg, params: IDEALParams, idx: int = 0
) -> np.ndarray:
    """IDEAL IVIM fitting recursive edition"""

    # TODO: dimension_steps should be sorted highest to lowest entry

    if idx:
        # Downsample image
        img = params.interpolate_img(nii_img.array, params.dimension_steps[idx])
        # Downsample segmentation
        try:
            seg = params.interpolate_seg(
                nii_seg.array,
                params.dimension_steps[idx],
                params.segmentation_threshold,
            )
        except AttributeError:
            seg = params.interpolate_seg(
                nii_seg.array, params.dimension_steps[idx], 0.025
            )
    else:
        # No sampling for last step
        img = nii_img.array
        seg = nii_seg.array

    # Recursion ahead
    if idx < params.dimension_steps.shape[1]:
        # Setup starting values, lower and upper bounds for fitting from previous/next step
        temp_parameters = fit_ideal_new(nii_img, nii_seg, params, idx + 1)
        x0 = params.interpolate_start_values_2d(
            temp_parameters, params.dimension_steps[idx]
        )
        lb = x0 * (1 - params.dimension_steps[idx])
        ub = x0 * (1 + params.dimension_steps[idx])
    else:
        # For last (1 x 1) Matrix the initial values are taken. This is the termination condition for the recursion
        x0 = params.boundaries["x0"]
        lb = params.boundaries["lb"]
        ub = params.boundaries["ub"]

    # Load pixel args with img and boundaries
    pixel_args: zip = params.get_pixel_args(img, seg, x0, lb, ub)

    # fit data
    print(f"Fitting: {params.dimension_steps[idx]}")
    fit_result = fit(
        params.fit_function, pixel_args, params.n_pools, multi_threading=False
    )

    fit_parameters = np.zeros(x0.shape)
    for key, var in fit_result:
        fit_parameters[key] = var

    return fit_parameters


# def fit_ideal(nii_img: Nii, params: IDEALParams, nii_seg: NiiSeg):
#     """IDEAL IVIM fitting based on Stabinska et al."""
#
#     # NOTE slice selection happens in original code here. if slices should be removed, do it in advance
#
#     # create partial for solver
#     fit_function = params.get_fit_function()
#
#     for step in params.dimension_steps:
#         # prepare output array
#         fitted_parameters = prepare_fit_output(
#             nii_seg.array, step, params.boundaries.x0
#         )
#
#         # NOTE Loop each resampling step -> Resample the whole volume and go to the next
#         if not np.array_equal(step, params.dimension_steps[-1]):
#             img_resampled, seg_resampled = resample_data(
#                 nii_img.array, nii_seg.array, step
#             )
#         else:
#             img_resampled = nii_img.array
#             seg_resampled = nii_seg.array
#
#         # NOTE Prepare Parameters
#         # TODO: merge if into prepare_parameters
#         if np.array_equal(step, params.dimension_steps[0]):
#             x0_resampled = np.zeros((1, 1, 1, len(params.boundaries.x0)))
#             x0_resampled[0, 0, 0, :] = params.boundaries.x0
#             lb_resampled = np.zeros((1, 1, 1, len(params.boundaries.lb)))
#             lb_resampled[0, 0, 0, :] = params.boundaries.lb
#             ub_resampled = np.zeros((1, 1, 1, len(params.boundaries.ub)))
#             ub_resampled[0, 0, 0, :] = params.boundaries.ub
#         else:
#             (
#                 x0_resampled,
#                 lb_resampled,
#                 ub_resampled,
#             ) = prepare_parameters(fitted_parameters, step, params.tol)
#
#         # NOTE instead of checking each slice for missing values check each calculated mask voxel and add only
#         # non-zero voxel to list
#
#         pixel_args = params.get_pixel_args(img_resampled, seg_resampled)
#
#         fit_results = fit_ideal(
#             fit_function, pixel_args, n_pools=4, multi_threading=False
#         )
#
#         # TODO extract fitted parameters
#
#         print("Test")


# def prepare_fit_output(seg: np.ndarray, step: np.ndarray, x0: np.ndarray):
#     new_shape = np.zeros((4, 1))
#     new_shape[:2, 0] = step
#     new_shape[2] = seg.shape[2]
#     new_shape[3] = len(x0)
#     return new_shape


# def resample_data(
#     img: np.ndarray,
#     seg: np.ndarray,
#     step_matrix_shape: np.ndarray,
#     resampling_lower_threshold: float | None = 0.025,
# ):
#     """ """
#     seg_resampled = np.zeros(
#         (step_matrix_shape[0], step_matrix_shape[1], seg.shape[2], seg.shape[3])
#     )
#     img_resampled = np.zeros(
#         (step_matrix_shape[0], step_matrix_shape[1], img.shape[2], img.shape[3])
#     )
#
#     # 2D processing
#     if step_matrix_shape.shape[0] == 2:
#         for slice_number in range(seg.shape[2]):
#             seg_slice = np.squeeze(seg[:, :, slice_number])
#
#             if step_matrix_shape[0] == 1:
#                 seg_resampled[:, :, slice_number] = np.ones((1, 1))
#             else:
#                 seg_resampled[:, :, slice_number] = ndimage.zoom(
#                     seg_slice,
#                     (
#                         step_matrix_shape[0] / seg_slice.shape[0],  # height
#                         step_matrix_shape[1] / seg_slice.shape[1],  # width
#                     ),
#                     order=1,
#                 )
#
#             for b_value in range(img.shape[3]):
#                 img_slice = img[:, :, slice_number, b_value]
#                 img_resampled[:, :, slice_number, b_value] = ndimage.zoom(
#                     img_slice,
#                     (
#                         step_matrix_shape[0] / img_slice.shape[0],  # height
#                         step_matrix_shape[1] / img_slice.shape[1],  # width
#                     ),
#                     order=1,
#                 )
#         seg_resampled = np.abs(seg_resampled)  # NOTE why?
#         # Threshold edges
#         seg_resampled[seg_resampled < resampling_lower_threshold] = 0
#     elif step_matrix_shape.shape[1] == 3:
#         print("3D data")
#     else:
#         # TODO Throw Error
#         print("Warning unknown step shape. Must be 2D or 3D")
#
#     return img_resampled, seg_resampled


# def prepare_parameters(
#     params: np.ndarray, step_matrix_shape: np.ndarray, tol: np.ndarray
# ):
#     x0_new = np.zeros(params.shape)
#     # calculate resampling factor
#     upscaling_factor = step_matrix_shape[0] / params.shape[0]
#     for parameter in range(params.shape[3]):
#         # fit_parameters should be a 4D array with fourth dimension containing the array of fitted parameters
#         for slice in range(params.shape[2]):
#             x0_new[:, :, slice, parameter] = ndimage.zoom(
#                 params[:, :, slice, parameter], upscaling_factor, order=1
#             )
#     lb_new = x0_new * (1 - tol)
#     ub_new = x0_new * (1 + tol)
#     return x0_new, lb_new, ub_new


def fit(
    fit_func: Callable,
    element_args: zip,
    n_pools: int,
    multi_threading: bool | None = True,
) -> list:
    """
    Args:
        fit_func:
        element_args:
        n_pools:
        multi_threading:

    Returns:
        list:
    """
    # TODO check for max cpu_count()
    if multi_threading:
        if n_pools != 0:
            with Pool(n_pools) as pool:
                results = pool.starmap(fit_func, element_args)
    else:
        results = []
        for element in element_args:
            results.append(fit_func(element[0], element[1], element[2]))

    return results
