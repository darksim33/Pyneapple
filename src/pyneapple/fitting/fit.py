from __future__ import annotations

import time

from ..utils.logger import logger
from pyneapple import Parameters
from radimgarray import RadImgArray, SegImgArray
from .multithreading import multithreader
from .. import IVIMParams, IVIMSegmentedParams
from .gpubridge import gpu_fitter


def fit_handler(params: Parameters, fit_args: zip, fit_type: str = None, **kwargs):
    """
    Handles fitting based on fit_type.

    Args:
        params (Parameters): Parameters for fitting. Can be IVIMParams,
            IVIMSegmentedParams, NNLSParams or NNLSCVParams.
        fit_args (zip): Fit arguments for fitting.
        fit_type (str): (Optional) Type of fitting to be used (single, multi, gpu). If
            not provided the fit_type from the parameters is used.
        kwargs (dict): Additional keyword arguments to pass to the fit function.
    """

    if not fit_type:
        fit_type = params.fit_type

    if fit_type in "multi":
        return multithreader(params.fit_function, fit_args, params.n_pools)
    elif fit_type in "single":
        return multithreader(params.fit_function, fit_args, None)
    elif fit_type in "gpu":
        if not isinstance(params, (IVIMParams, IVIMSegmentedParams)):
            error_msg = "GPU fitting only is available for IVIM fitting atm."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return gpu_fitter(fit_args, params, **kwargs)
    else:
        error_msg = f"Unsupported or unset fit_type ({fit_type})."
        logger.error(error_msg)
        raise ValueError(error_msg)


def fit_pixel_wise(
        img: RadImgArray,
        seg: SegImgArray,
        params: Parameters,
        fit_type: str = None,
        **kwargs
) -> list:
    """Fits every pixel inside the segmentation individually.

    Args:
        params (Parameters): Parameter object with fitting parameters.
        img (RadImgArray): RadImgArray object with image data.
        seg (SegImgArray): SegImgArray object with segmentation data.
        fit_type (str): (Optional) Type of fitting to be used (single, multi, gpu).
        kwargs (dict): Additional keyword arguments to pass to the fit function.
    """
    results = list()
    if params.json is not None and params.b_values is not None:
        logger.info("Fitting pixel wise...")
        start_time = time.time()
        pixel_args = params.get_pixel_args(img, seg)
        results = fit_handler(params, pixel_args, fit_type, **kwargs)
        logger.info(f"Pixel-wise fitting time: {round(time.time() - start_time, 2)}s")
    else:
        error_msg = "No valid Parameter Set for fitting selected!"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return results


def fit_segmentation_wise(
        img: RadImgArray,
        seg: SegImgArray,
        params: Parameters,
) -> list:
    """Fits mean signal of segmentation(s), computed of all pixels signals."""

    results = list()
    if params.json is not None and params.b_values is not None:
        logger.info("Fitting segmentation wise...")
        start_time = time.time()
        for seg_number in seg.seg_values:
            # get mean pixel signal
            seg_args = params.get_seg_args(img, seg, seg_number)
            # fit mean signal
            seg_results = fit_handler(params, seg_args, "single")

            # Save result of mean signal for every pixel of each seg
            results.append((seg_number, seg_results[0][1]))

        logger.info(f"Segmentation-wise fitting time: {round(time.time() - start_time, 2)}s")
    else:
        error_msg = "No valid Parameter Set for fitting selected!"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return results


def fit_ivim_segmented(
        img: RadImgArray,
        seg: SegImgArray,
        params: IVIMSegmentedParams,
        fit_type: str = None,
        debug: bool = False,
        **kwargs
) -> tuple[list, list]:
    """IVIM Segmented Fitting Interface.
    Args:
        params (IVIMSegmentedParams): Parameter object with fitting parameters.
        img (RadImgArray): RadImgArray object with image data.
        seg (SegImgArray): SegImgArray object with segmentation data.
        fit_type (str): (Optional) Type of fitting to be used (single, multi, gpu).
        debug (bool): If True, debug output is printed.
        kwargs (dict): Additional keyword arguments to pass to the fit function.
    """
    start_time = time.time()
    logger.info("Fitting first component for segmented IVIM model...")
    # Get Pixel Args for first Fit
    pixel_args = params.get_pixel_args_fixed(img, seg)

    # Run First Fitting
    results = fit_handler(params.params_fixed, pixel_args, fit_type)
    fixed_component = params.get_fixed_fit_results(results)

    pixel_args = params.get_pixel_args(img, seg, *fixed_component)

    # Run Second Fitting
    logger.info("Fitting all remaining components for segmented IVIM model...")
    results = fit_handler(params, pixel_args, fit_type, **kwargs)
    logger.info(f"Pixel-wise segmented fitting time: {round(time.time() - start_time, 2)}s")
    return fixed_component, results


def fit_IDEAL(
        img: RadImgArray,
        seg: SegImgArray,
        params: IDEALParams,
        multi_threading: bool = False,
        debug: bool = False,
):
    """IDEAL Fitting Interface.
    Args:
        multi_threading (bool): If True, multi-threading is used.
        debug (bool): If True, debug output is printed.
    """
    start_time = time.time()
    logger.info(f"The initial image size is {img.shape[0:4]}.")
    fit_results = fit_IDEAL(img, seg, params, multi_threading, debug)
    # results.eval_results(fit_results)
    logger.info(f"IDEAL fitting time:{round(time.time() - start_time, 2)}s")
    return fit_results
