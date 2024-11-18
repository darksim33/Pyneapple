from __future__ import annotations

import time

from pyneapple import Parameters
from radimgarray import RadImgArray, SegImgArray
from .multithreading import multithreader
from .. import IVIMSegmentedParams, IDEALParams


def fit_pixel_wise(
    img: RadImgArray,
    seg: SegImgArray,
    params: Parameters,
    multi_threading: bool | None = True,
) -> list:
    """Fits every pixel inside the segmentation individually.

    Args:
        params (Parameters): Parameter object with fitting parameters.
        img (RadImgArray): RadImgArray object with image data.
        seg (SegImgArray): SegImgArray object with segmentation data.
        multi_threading (bool | None): If True, multi-threading is used.
    """
    results = list()
    if params.json is not None and params.b_values is not None:
        print("Fitting pixel wise...")
        start_time = time.time()
        pixel_args = params.get_pixel_args(img, seg)
        results = multithreader(
            params.fit_function,
            pixel_args,
            params.n_pools if multi_threading else None,
        )
        print(f"Pixel-wise fitting time: {round(time.time() - start_time, 2)}s")
    else:
        ValueError("No valid Parameter Set for fitting selected!")
    return results


def fit_segmentation_wise(
    img: RadImgArray,
    seg: SegImgArray,
    params: Parameters,
) -> list:
    """Fits mean signal of segmentation(s), computed of all pixels signals."""

    results = list()
    if params.json is not None and params.b_values is not None:
        print("Fitting segmentation wise...")
        start_time = time.time()
        for seg_number in seg.seg_values:
            # get mean pixel signal
            seg_args = params.get_seg_args(img, seg, seg_number)
            # fit mean signal
            seg_results = multithreader(
                params.fit_function,
                seg_args,
                n_pools=None,
            )

            # Save result of mean signal for every pixel of each seg
            # for pixel in seg.get_seg_indices(seg_number):
            #     results.append((pixel, seg_results[0][1]))
            results.append((seg_number, seg_results[0][1]))

        # TODO: seg.seg_indices now returns an list of tuples
        print(f"Segmentation-wise fitting time: {round(time.time() - start_time, 2)}s")
    else:
        ValueError("No valid Parameter Set for fitting selected!")
    return results


def fit_ivim_segmented(
    img: RadImgArray,
    seg: SegImgArray,
    params: IVIMSegmentedParams,
    multi_threading: bool = False,
    debug: bool = False,
) -> tuple[list, list]:
    """IVIM Segmented Fitting Interface.
    Args:
        params (IVIMSegmentedParams): Parameter object with fitting parameters.
        img (RadImgArray): RadImgArray object with image data.
        seg (SegImgArray): SegImgArray object with segmentation data.
        multi_threading (bool): If True, multi-threading is used.
        debug (bool): If True, debug output is printed.
    """
    start_time = time.time()
    print("Fitting first component for segmented IVIM model...")
    # Get Pixel Args for first Fit
    pixel_args = params.get_pixel_args_fixed(img, seg)

    # Run First Fitting
    results = multithreader(
        params.params_fixed.fit_function,
        pixel_args,
        params.n_pools if multi_threading else None,
    )
    fixed_component = params.get_fixed_fit_results(results)

    pixel_args = params.get_pixel_args(img, seg, *fixed_component)

    # Run Second Fitting
    print("Fitting all remaining components for segmented IVIM model...")
    results = multithreader(
        params.fit_function,
        pixel_args,
        params.n_pools if multi_threading else None,
    )
    print(f"Pixel-wise segmented fitting time: {round(time.time() - start_time, 2)}s")
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
    print(f"The initial image size is {img.shape[0:4]}.")
    fit_results = fit_IDEAL(img, seg, params, multi_threading, debug)
    # results.eval_results(fit_results)
    print(f"IDEAL fitting time:{round(time.time() - start_time, 2)}s")
    return fit_results
