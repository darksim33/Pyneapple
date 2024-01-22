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
from .parameters import IVIMParams, Results
from src.utils import Nii, NiiSeg, NiiFit

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
                args: np.ndarray,
                lb: np.ndarray,
                ub: np.ndarray,
                b_values: np.ndarray,
                n_components: int,
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
    tolerance: np.ndarray
        ideal adjustment tolerance for each parameter
    dimension_steps: np.ndarray
        down-sampling steps for fitting

    """

    def __init__(
            self,
            params_json: Path | str = None,
    ):
        self.tolerance = None
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
            n_components=self.n_components,
            max_iter=self.max_iter,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        self._fit_function = method

    @property
    def fit_model(self):
        return self._fit_model(n_components=self.n_components)

    @fit_model.setter
    def fit_model(self, method: Callable):
        self._fit_model = method

    @property
    def boundaries(self):
        return self._boundaries

    @boundaries.setter
    def boundaries(self, values: dict):
        # Make sure every entry in the boundaries is a np.array
        for key, value in values.items():
            if isinstance(value, list):
                values[key] = np.array(value)
        self._boundaries = values

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

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
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

    def load_from_json(self, params_json: str | Path | None = None):
        """Loads json files for IDEAL IVIM processing."""
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
            self, boundary: np.ndarray, matrix_shape: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate starting values for the given boundary.

        boundary: np.ndarray of shape(x, y, z, n_variables).
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        # if boundary.shape[0:1] < matrix_shape:
        new_boundary = np.zeros(
            (matrix_shape[0], matrix_shape[1], boundary.shape[2], boundary.shape[3])
        )
        # interpolate slice by slice and...
        plane: np.ndarray
        for idx_slice, plane in enumerate(boundary.transpose(2, 0, 1, 3)):
            # ... fitting variable by variable
            variables: np.ndarray
            for idx_variable, variables in enumerate(plane.transpose(2, 0, 1)):
                new_boundary[:, :, idx_slice, idx_variable] = self.interpolate_array(
                    variables, matrix_shape
                )
        return new_boundary

    def interpolate_img(
            self,
            img: np.ndarray,
            matrix_shape: np.ndarray | list | tuple,
            multithreading: bool = False,
    ) -> np.ndarray:
        """
        Interpolate image to desired size in 2D.

        img: np.ndarray of shape(x, y, z, n_bvalues)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        new_image = np.zeros(
            (matrix_shape[0], matrix_shape[1], img.shape[2], img.shape[3])
        )
        if not multithreading:
            # interpolate slice by slice...
            plane: np.ndarray
            for idx_slice, plane in enumerate(img.transpose(2, 0, 1, 3)):
                # ... and b-value/decay point by point
                decay: np.ndarray
                for idx_decay, decay in enumerate(plane.transpose(2, 0, 1)):
                    new_image[:, :, idx_slice, idx_decay] = self.interpolate_array(
                        decay, matrix_shape
                    )
        else:
            args_list = [
                (idx_slice, plane, matrix_shape)
                for idx_slice, plane in enumerate(img.transpose(2, 0, 1, 3))
            ]
        return new_image

    def interpolate_seg(
            self,
            seg: np.ndarray,
            matrix_shape: np.ndarray | list | tuple,
            threshold: float,
            multithreading: bool = False,
            n_pools: int | None = 4,
    ) -> np.ndarray:
        """
        Interpolate segmentation to desired size in 2D and apply threshold.

        seg: np.ndarray of shape(x, y, z)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        seg_new = np.zeros((matrix_shape[0], matrix_shape[1], seg.shape[2]))
        if not multithreading:
            # interpolate slice by slice
            plane: np.ndarray
            for idx_slice, plane in enumerate(seg.transpose(2, 0, 1, 3)):
                seg_new[:, :, idx_slice] = self.interpolate_array(plane, matrix_shape)
        else:
            args_list = zip(
                (idx_slice for idx_slice, _ in enumerate(seg.transpose(2, 0, 1, 3))),
                (plane for plane in seg.transpose(2, 0, 1, 3))
            )
            if n_pools != 0:
                with Pool(n_pools) as pool:
                    results = pool.starmap(
                        partial(self.interpolate_array_multithreading, matrix_shape=matrix_shape),
                        args_list,
                    )
            for element in results:
                seg_new[:, :, element[0]] = element[1]

        # Make sure Segmentation is binary
        seg_new[seg_new < threshold] = 0
        seg_new[seg_new > threshold] = 1

        # Check seg size. Needs to be M x N x Z x 1
        while len(seg_new.shape) < 4:
            seg_new = np.expand_dims(seg_new, axis=len(seg_new.shape))
        return seg_new

    @staticmethod
    def interpolate_array(array: np.ndarray, matrix_shape: np.ndarray):
        """Interpolate 2D array to new shape."""

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

    @staticmethod
    def interpolate_array_multithreading(
            idx: tuple, array: np.ndarray, matrix_shape: np.ndarray
    ):
        array = IDEALParams.interpolate_array(array, matrix_shape)
        return idx, array

    def eval_fitting_results(self, results: np.ndarray, seg: NiiSeg) -> Results:
        """
        Evaluate fitting results for the IDEAL method.

        Parameters
        ----------
            results
                Pass the results of the fitting process to this function
            seg: NiiSeg
                Get the shape of the spectrum array
        """
        coordinates = seg.get_seg_coordinates("nonzero")
        # results_zip = list(zip(coordinates, results[coordinates]))
        results_zip = zip(
            (coord for coord in coordinates), (results[coord] for coord in coordinates)
        )
        fit_results = super().eval_fitting_results(results_zip, seg)
        return fit_results


def fit_ideal_new(
        nii_img: Nii,
        nii_seg: NiiSeg,
        params: IDEALParams,
        idx: int = 0,
        multithreading: bool = False,
        debug: bool = False,
) -> np.ndarray:
    """
    IDEAL IVIM fitting recursive edition.
    :param nii_img: Nii image 4D containing the decay in the fourth dimension
    :param nii_seg: Nii segmentation 3D with an empty fourth dimension
    :param params: IDEAL parameters which might be removed?
    :param idx: Current iteration index
    :multithreading: Enables multithreading or not
    :param debug: Debugging option
    """

    # TODO: dimension_steps should be sorted highest to lowest entry

    print(f"Prepare Image and Segmentation for step {params.dimension_steps[idx]}")
    if idx:
        # Downsample image
        img = params.interpolate_img(nii_img.array, params.dimension_steps[idx])
        # Downsample segmentation.
        seg = params.interpolate_seg(
            nii_seg.array,
            params.dimension_steps[idx],
            params.segmentation_threshold,
            multithreading=multithreading,
        )
        # Check if down sampled segmentation is valid. If the resampled matrix is empty the whole matrix is used
        if not seg.max():
            seg = np.ones(seg.shape)

        if debug:
            Nii().from_array(img).save(
                "data/ideal/img_" + str(idx) + ".nii.gz"
            )
            Nii().from_array(seg).save(
                "data/ideal/seg_" + str(idx) + ".nii.gz"
            )
    else:
        # No sampling for last step/ fitting of the actual image
        img = nii_img.array
        seg = nii_seg.array

    # Recursion ahead
    if idx < params.dimension_steps.shape[0] - 1:
        # Setup starting values, lower and upper bounds for fitting from previous/next step
        temp_parameters = fit_ideal_new(nii_img, nii_seg, params, idx + 1, multithreading=multithreading, debug=debug)

        # if the lowest matrix size was reached (1x1 for the default case)
        # the matrix for the next step is set manually cause interpolation
        # from 1x1 is not possible with the current implementation
        if temp_parameters.shape[0] == 1:
            x0 = np.repeat(temp_parameters, params.dimension_steps[idx, 0], axis=0)
            x0 = np.repeat(x0, params.dimension_steps[idx, 0], axis=1)
        else:
            # for all other steps the interpolated values are used
            x0 = params.interpolate_start_values_2d(
                temp_parameters, params.dimension_steps[idx]
            )
        lb = x0 * (1 - params.tolerance)
        ub = x0 * (1 + params.tolerance)
    else:
        # For last (1 x 1) Matrix the initial values are taken. This is the termination condition for the recursion
        x0 = np.zeros((1, 1, seg.shape[2], params.boundaries["x0"].size))
        x0[::, :] = params.boundaries["x0"]
        lb = np.zeros((1, 1, seg.shape[2], params.boundaries["x0"].size))
        lb[::, :] = params.boundaries["lb"]
        ub = np.zeros((1, 1, seg.shape[2], params.boundaries["ub"].size))
        ub[::, :] = params.boundaries["ub"]

    # Load pixel args with img and boundaries
    pixel_args: zip = params.get_pixel_args(img, seg, x0, lb, ub)

    # fit data
    print(f"Fitting: {params.dimension_steps[idx]}")
    fit_result = fit_agent(
        params.fit_function, pixel_args, params.n_pools, multi_threading=multithreading
    )

    # transfer fitting results from dictionary to matrix
    fit_parameters = np.zeros(x0.shape)
    for key, var in fit_result:
        fit_parameters[key] = var

    # if debug:
    #     NiiFit(n_components=params.n_components).from_array(fit_parameters).save(
    #         "data/ideal/fit_" + str(idx) + ".nii.gz"
    #     )
    return fit_parameters


def fit_agent(
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
            results.append(
                fit_func(element[0], element[1], element[2], element[3], element[4])
            )

    return results
