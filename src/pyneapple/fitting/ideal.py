"""
IDEAL based down sampling and fitting approach. Utilizes image down sampling to get more
suitable fitting parameters and there by improve the fitting results.

Methods:
    fit_IDEAL(nii_img: Nii, nii_seg: NiiSeg, params: Parameters | IDEALParams,
        multi_threading: bool = False, debug: bool = False)
        IDEAL IVIM fitting job.
    setup(nii_img: Nii, nii_seg: NiiSeg, **kwargs)
        Setup for the IDEAL fitting.
    fit_recursive(nii_img: Nii, nii_seg: NiiSeg, params: Parameters | IDEALParams,
        idx: int = 0, multi_threading: bool = False, debug: bool = False)
        IDEAL IVIM fitting recursive edition.
"""

from __future__ import annotations

import numpy as np

from .. import Parameters, IDEALParams
from nifti import Nii, NiiSeg
from ..utils import processing
from .multithreading import multithreader, sort_fit_array


def fit_IDEAL(
    nii_img: Nii,
    nii_seg: NiiSeg,
    params: Parameters | IDEALParams,
    multi_threading: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """IDEAL IVIM fitting job.

    Args:
        nii_img (Nii): Nii image 4D containing the decay in the fourth dimension.
        nii_seg (NiiSeg): Nii segmentation 3D with an empty fourth dimension.
        params (Parameters | IDEALParams): IDEAL parameters which might be removed?
        multi_threading (bool): Enables multithreading or not.
        debug (bool): Debugging option.

    Returns:
        fit_result (np.ndarray): Fitting result.
    """
    nii_img, nii_seg = setup(nii_img, nii_seg, params=params, crop=True)
    fit_result = fit_recursive(
        nii_img=nii_img,
        nii_seg=nii_seg,
        params=params,
        idx=0,
        multi_threading=multi_threading,
        debug=debug,
    )
    return fit_result


def setup(nii_img: Nii, nii_seg: NiiSeg, **kwargs):
    """Setup for the IDEAL fitting.

    Args:
        nii_img (Nii): Nii image 4D containing the decay in the fourth dimension.
        nii_seg (NiiSeg): Nii segmentation 3D with an empty fourth dimension.
        **kwargs: Arbitrary keyword arguments.
    Returns:
        nii_img (Nii): image with applied segmentation (cut)
        nii_seg (NiiSeg): segmentation with only one segmentation index left
    """
    if kwargs.get("crop", False):
        # Make sure the segmentation only contains values of 0 and 1
        seg = nii_seg.array.copy()
        seg[seg > 0] = 1
        new_img = processing.merge_nii_images(nii_img, NiiSeg().from_array(seg))
        nii_img = new_img
        print("Cropping image.")
    return nii_img, nii_seg
    # TODO: dimension_steps should be sorted highest to lowest entry
    # apply mask to image to reduce load by segmentation resampling
    # check if matrix is squared and if final dimension is fitting to actual size.


def fit_recursive(
    nii_img: Nii,
    nii_seg: NiiSeg,
    params: Parameters | IDEALParams,
    idx: int = 0,
    multi_threading: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """IDEAL IVIM fitting recursive edition.

    Args:
        nii_img (Nii): Nii image 4D containing the decay in the fourth dimension.
        nii_seg (NiiSeg): Nii segmentation 3D with an empty fourth dimension.
        params (Parameters | IDEALParams): IDEAL parameters which might be removed?
        idx (int): Current iteration index.
        multi_threading (bool): Enables multithreading or not.
        debug (bool): Debugging option.

    Returns:
        fit_parameters (np.ndarray): Fitting parameters for next step.
    """

    # TODO: NiiSeg should be omitted. Maybe args should be parsed
    #       from segmented image (:,:,:,0)

    print(f"Prepare Image and Segmentation for step {params.dimension_steps[idx]}")
    if idx:
        # Down-sample image
        img = params.interpolate_img(
            nii_img.array,
            params.dimension_steps[idx],
            # n_pools=params.n_pools if multi_threading else None,
            n_pools=None,
        )
        # Down-sample segmentation.
        seg = params.interpolate_seg(
            nii_seg.array,
            params.dimension_steps[idx],
            params.segmentation_threshold,
            # n_pools=params.n_pools if multi_threading else None,
            n_pools=None,
        )
        # Check if down sampled segmentation is valid. If the resampled matrix is
        # empty the whole matrix is used
        if not seg.max():
            seg = np.ones(seg.shape)

        if debug:
            Nii().from_array(img).save("data/IDEAL/img_" + str(idx) + ".nii.gz")
            Nii().from_array(seg).save("data/IDEAL/seg_" + str(idx) + ".nii.gz")
    else:
        # No sampling for last step/ fitting of the actual image
        img = nii_img.array
        seg = nii_seg.array

    # Recursion ahead
    if idx < params.dimension_steps.shape[0] - 1:
        # Setup starting values, lower and upper bounds for fitting
        # from previous/next step
        temp_parameters = fit_recursive(
            nii_img,
            nii_seg,
            params,
            idx + 1,
            multi_threading=multi_threading,
            debug=debug,
        )

        # if the lowest matrix size was reached (1x1 for the default case)
        # the matrix for the next step is set manually cause interpolation
        # from 1x1 is not possible with the current implementation
        if temp_parameters.shape[0] == 1:
            x0 = np.repeat(temp_parameters, params.dimension_steps[idx, 0], axis=0)
            x0 = np.repeat(x0, params.dimension_steps[idx, 0], axis=1)
        else:
            # for all other steps the interpolated values are used
            x0 = params.interpolate_start_values_2d(
                temp_parameters,
                params.dimension_steps[idx],
                n_pools=params.n_pools if multi_threading else None,
            )
        lb = x0 * (1 - params.tolerance)
        ub = x0 * (1 + params.tolerance)
    else:
        # For last (1 x 1) Matrix the initial values are taken. This is the
        # termination condition for the recursion
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
    fit_result = multithreader(
        params.fit_function,
        pixel_args,
        n_pools=params.n_pools if multi_threading else None,
    )
    fit_parameters = np.zeros(x0.shape)
    # transfer fitting results from dictionary to matrix
    fit_parameters[:] = sort_fit_array(fit_result, fit_parameters)

    # TODO: implement fit saving for debugging
    # if debug:
    #     NiiFit(n_components=params.n_components).from_array(fit_parameters).save(
    #         "data/IDEAL/fit_" + str(idx) + ".nii.gz"
    #     )
    if debug:
        fit_results = params.eval_fitting_results(
            fit_parameters, NiiSeg().from_array(seg)
        )
        fit_results.save_fitted_parameters_to_nii(
            file_path="data/IDEAL/fit_" + str(idx) + ".nii",
            shape=img.shape,
            dtype=float,
        )
    return fit_parameters
