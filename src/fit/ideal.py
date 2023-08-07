import numpy as np
from scipy import ndimage

from src.utils import Nii, Nii_seg
from .fit import *
from .parameters import *
from .model import Model

class ideal_fitting(object):
    class IDEALParams(Parameters):
        """
        IDEAL fitting Parameter class

        ...

        Attributes
        ----------
        model: FitModel
            FitModel used in IDEAL approach
            The default model is the triexponential Model with starting values optimized for kidney
            Parameters are as follows: D_fast, D_interm, D_slow, f_fast, f_interm, (S0)
            ! For theses Models the last fraction is typicaly calculated from the sum of fraction
        lb: np.ndarray
            Lower fitting boundaries
        ub: np.ndarray
            upper fitting boundaries
        x0: np.ndarray
            starting values
        tol: np.ndarray
            ideal adjustment tolerance for each parameter
        dimension_steps: np.ndarray
            downsampling steps for fitting

        """

        def __init__(
            self,
            model: Model | None = None,  # triexponential Mode
            b_values: np.ndarray | None = np.array([]),
            lb: np.ndarray
            | None = np.array(
                [
                    0.01,  # D_fast
                    0.003,  # D_intermediate
                    0.0011,  # D_slow
                    0.01,  # f_fast
                    0.1,  # f_interm
                ]
            ),
            ub: np.ndarray
            | None = np.array(
                [
                    0.5,  # D_fast
                    0.01,  # D_interm
                    0.003,  # D_slow
                    0.7,  # f_fast
                    0.7,  # f_interm
                ]
            ),
            x0: np.ndarray
            | None = np.array(
                [
                    0.1,  # D_fast
                    0.005,  # D_interm
                    0.0015,  # D_slow
                    0.1,  # f_fast
                    0.2,  # f_interm
                ]
            ),
            tol: np.ndarray
            | None = np.array(
                [0.2, 0.2, 0.2, 0.1, 0.1]
            ),  # one tolerance for each parameter
            dimension_steps: np.ndarray
            | None = np.array(
                [
                    [1, 1],
                    [2, 2],
                    [4, 4],
                    [8, 8],
                    [16, 16],
                    [32, 32],
                    [64, 64],
                    [96, 96],
                    [128, 128],
                    [156, 156],
                    [176, 176],
                ]
            ),  # height, width, depth
            max_iter: int | None = 600,
        ):
            super().__init__(model, b_values=b_values)
            self.model = model
            self.boundaries.lb = lb
            self.boundaries.ub = ub
            self.boundaries.x0 = x0
            self.tol = tol
            # self.dimensions = dimension_steps.shape[1]
            self.dimension_steps = dimension_steps
            self.max_iter = max_iter

        def get_pixel_args(
            self,
            img: np.ndarray,
            seg: np.ndarray,
            x0: np.ndarray,
            lb: np.ndarray,
            ub: np.ndarray,
        ):
            # Behaves the same way as the original parent funktion with the difference that instead of Nii objects np.ndarrays are passed
            # Also needs to pack all aditional fitting parameters x0, lb, ub
            pixel_args = zip(
                ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
                (
                    img[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
                ),
                (
                    x0[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
                ),
                (
                    lb[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
                ),
                (
                    ub[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
                ),
            )
            return pixel_args

        def get_fit_function(self):
            return partial(
                self.model,
                b_values=self.get_basis(),
                max_iter=self.max_iter,
            )

    def fit_ideal(nii_img: Nii, parameters: IDEALParams, nii_seg: Nii_seg, debug: bool):
        """
        IDEAL IVIM fitting based on Stabinska et al.
        """

        # NOTE slice selection happens in original code here. if slices should be removed, do it in advance

        # create partial for solver
        fit_function = parameters.get_fit_function()

        for step in parameters.dimension_steps:
            # prepare output array
            fitted_parameters = ideal_fitting.prepare_fit_output(
                nii_seg.array, step, parameters.boundaries.x0
            )

            # NOTE Loop each resampling step -> Resample the whole volume and go to the next
            if step != parameters.dimension_steps[-1]:
                img_resampled, seg_resampled = ideal_fitting.resample_data(
                    nii_img.array, nii_seg.array, step
                )
            else:
                img_resampled = nii_img.array
                seg_resampled = nii_seg.array

            # NOTE Prepare Parameters
            if step == parameters.dimension_steps[0]:
                x0_resampled = parameters.boundaries.x0
                lb_resampled = parameters.boundaries.lb
                ub_resampled = parameters.boundaries.ub
            else:
                (
                    x0_resampled,
                    lb_resampled,
                    ub_resampled,
                ) = ideal_fitting.prepare_parameters(
                    fitted_parameters, step, parameters.tol
                )

            # NOTE instead of checking each slice for missing values check each calculated mask voxel and add only non-zero voxel to list
            pixel_args = parameters.get_pixel_args(
                img_resampled, seg_resampled, x0_resampled, lb_resampled, ub_resampled
            )

            fit_results = fit(fit_function, pixel_args)

            # TODO extract fitted parameters

            print("Test")

    def prepare_fit_output(seg: np.ndarray, step: np.ndarray, x0: np.ndarray):
        new_shape = step
        new_shape[2] = seg.shape[2]
        new_shape[3] = len(x0)
        return np.zeros(new_shape)

    def resample_data(
        img: np.ndarray,
        seg: np.ndarray,
        step_matrix_shape: np.ndarray,
        resampling_lower_threshold: float | None = 0.025,
    ):
        """ """
        seg_resampled = np.zeros(seg.shape)
        img_resampled = np.zeros(img.shape)

        # 2D processing
        if step_matrix_shape.shape[1] == 2:
            for slice in range(seg.shape[2]):
                seg_slice = np.squeeze(seg[:, :, slice])
                img_slice = np.squeeze(img[:, :, slice, :])

                seg_resampled[:, :, slice] = ndimage.zoom(
                    seg_slice,
                    (
                        step_matrix_shape[0] * seg_slice.shape[0],  # height
                        step_matrix_shape[1] * seg_slice.shape[1],  # width
                    ),
                    order=1,
                )

                for b_value in range(img.shape[3]):
                    img_resampled[:, :, slice, b_value] = ndimage.zoom(
                        img_slice,
                        (
                            step_matrix_shape[0] * img_slice.shape[0],  # height
                            step_matrix_shape[1] * img_slice.shape[1],  # width
                        ),
                        order=1,
                    )
            seg_resampled = np.abs(seg_resampled)  # NOTE why?
            # Threshold edeges
            seg_resampled[seg_resampled < resampling_lower_threshold] = 0
        elif step_matrix_shape.shape[1] == 3:
            print("3D data")
        else:
            # TODO Throw Error
            print("Warning unknown step shape. Must be 2D or 3D")

        return seg_resampled, img_resampled

    def prepare_parameters(
        parameters: np.ndarray, step_matrix_shape: np.ndarray, tol: np.ndarray
    ):
        x0_new = np.zeros(parameters.shape)
        # calculate resampling factor
        upscaling_factor = step_matrix_shape[0] / parameters.shape[0]
        for parameter in range(parameters.shape[3]):
            # fit_parameters should be a 4D array with fourth dimension containing the array of fitted parameters
            for slice in range(parameters.shape[2]):
                x0_new[:, :, slice, parameter] = ndimage.zoom(
                    parameters[:, :, slice, parameter], upscaling_factor, order=1
                )
        lb_new = x0_new * (1 - tol)
        ub_new = x0_new * (1 + tol)
        return x0_new, lb_new, ub_new
