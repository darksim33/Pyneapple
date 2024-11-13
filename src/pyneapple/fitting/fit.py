"""
Module combining fitting methods for pixel- and segmentation-wise fitting.
Main interface for fitting methods.

Classes:
    FitData: Fitting class for (multithreaded) pixel- and segmentation-wise fitting.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from radimgarray import RadImgArray, SegImgArray
from .. import (
    Parameters,
    IVIMParams,
    IVIMSegmentedParams,
    NNLSParams,
    NNLSCVParams,
    IDEALParams,
    NNLSResults,
    IVIMResults,
)
from .multithreading import multithreader
from .ideal import fit_IDEAL
from ..results.results import Results
from .. import IVIMResults, IVIMSegmentedResults, NNLSResults


class FitData:
    """Fitting class for (multithreaded) pixel- and segmentation-wise fitting.

    Attributes:
        model_name (str): Model name for fitting.
        params_json (str, Path): Path to json file with fitting parameters.
        img (RadImgArray): RadImgArray object with image data (4D).
        seg (SegImgArray): SegImgArray object with segmentation data (4D).

    Methods:
        fit_pixel_wise(multi_threading: bool | None = True)
            Fits every pixel inside the segmentation individually.
            Multi-threading possible to boost performance.
        fit_segmentation_wise()
            Fits mean signal of segmentation(s).
    """

    def __init__(
        self,
        model: str | None = None,
        params_json: str | Path | None = None,
        img: RadImgArray | None = None,  # Maybe Change signature later
        seg: SegImgArray | None = None,
    ):
        """Initializes Fitting Class.

        Args:
            model (str): Model name for fitting.
            params_json (str, Path): Path to json file with fitting parameters.
            img (RadImgArray): RadImgArray object with image data.
            seg (SegImgArray): SegImgArray object with segmentation data.
        """
        self.model_name = model
        self.params_json = params_json
        self.img = img
        self.seg = seg
        if model == "NNLS":
            self.params = NNLSParams(params_json)
            self.results = NNLSResults(self.params)
        elif model == "NNLSCV":
            self.params = NNLSCVParams(params_json)
            self.results = NNLSResults(self.params)
        elif model == "IVIM":
            self.params = IVIMParams(params_json)
            self.results = IVIMResults(self.params)
        elif model == "IVIMSegmented":
            self.params = IVIMSegmentedParams(params_json)
            self.results = IVIMSegmentedResults(self.params)
        elif model == "IDEAL":
            self.params = IDEALParams(params_json)
            self.results = IVIMResults(self.params)
        else:
            self.params = Parameters(params_json)
            # print("Warning: No valid Fitting Method selected")
        self.flags = dict()
        self.set_default_flags()

    def set_default_flags(self):
        """Sets default flags for fitting class."""
        self.flags["did_fit"] = False

    def reset(self):
        """Resets the fitting class."""
        self.model_name = None
        self.img = None
        self.seg = None
        self.params = Parameters()
        self.results = Results(self.params)
        self.set_default_flags()

    def fit_pixel_wise(self, multi_threading: bool | None = True):
        """Fits every pixel inside the segmentation individually.

        Args:
            multi_threading (bool | None): If True, multi-threading is used.
        """
        if self.params.json is not None and self.params.b_values is not None:
            print(f"Fitting {self.model_name} pixel wise...")
            start_time = time.time()
            pixel_args = self.params.get_pixel_args(self.img, self.seg)

            results = multithreader(
                self.params.fit_function,
                pixel_args,
                self.params.n_pools if multi_threading else None,
            )

            self.results.eval_results(results)
            print(f"Pixel-wise fitting time: {round(time.time() - start_time, 2)}s")
        else:
            ValueError("No valid Parameter Set for fitting selected!")

    def fit_segmentation_wise(self):
        """Fits mean signal of segmentation(s), computed of all pixels signals."""
        if self.img is None or self.seg is None:
            raise ValueError("No valid data for fitting selected!")

        if self.params.json is not None and self.params.b_values is not None:
            print(f"Fitting {self.model_name} segmentation wise...")
            start_time = time.time()
            results = list()
            seg_indices = dict()
            for seg_number in self.seg.seg_values:
                # get mean pixel signal
                seg_args = self.params.get_seg_args(self.img, self.seg, seg_number)
                # fit mean signal
                seg_results = multithreader(
                    self.params.fit_function,
                    seg_args,
                    n_pools=None,  # self.params.n_pools,
                )

                # Save result of mean signal for every pixel of each seg
                # for pixel in self.seg.get_seg_indices(seg_number):
                #     results.append((pixel, seg_results[0][1]))
                results.append((seg_number, seg_results[0][1]))
                indices = np.squeeze(self.seg, axis=3).get_seg_indices(seg_number)
                seg_indices.update(
                    {
                        key: value
                        for (key, value) in zip(
                            indices, [seg_number * 1] * len(indices)
                        )
                    }
                )

            # TODO: seg.seg_indices now returns an list of tuples
            self.results.set_segmentation_wise(seg_indices)

            self.results.eval_results(results)

            print(
                f"Segmentation-wise fitting time: {round(time.time() - start_time, 2)}s"
            )
        else:
            ValueError("No valid Parameter Set for fitting selected!")

    def fit_IDEAL(self, multi_threading: bool = False, debug: bool = False):
        """IDEAL Fitting Interface.
        Args:
            multi_threading (bool): If True, multi-threading is used.
            debug (bool): If True, debug output is printed.
        """
        start_time = time.time()
        if not self.model_name == "IDEAL":
            raise AttributeError("Wrong model name!")
        print(f"The initial image size is {self.img.shape[0:4]}.")
        fit_results = fit_IDEAL(self.img, self.seg, self.params, multi_threading, debug)
        self.results.eval_results(fit_results)
        print(f"IDEAL fitting time:{round(time.time() - start_time, 2)}s")

    def fit_ivim_segmented(self, multi_threading: bool = False, debug: bool = False):
        """IVIM Segmented Fitting Interface.
        Args:
            multi_threading (bool): If True, multi-threading is used.
            debug (bool): If True, debug output is printed.
        """
        self.params: IVIMSegmentedParams
        start_time = time.time()
        if not self.model_name == "IVIMSegmented":
            raise AttributeError("Wrong model name!")
        print("Fitting first component for segmented IVIM model...")

        # Get Pixel Args for first Fit
        pixel_args = self.params.get_pixel_args_fixed(self.img, self.seg)

        # Run First Fitting
        results = multithreader(
            self.params.params_fixed.fit_function,
            pixel_args,
            self.params.n_pools if multi_threading else None,
        )
        fixed_component = self.params.get_fixed_fit_results(results)

        pixel_args = self.params.get_pixel_args(self.img, self.seg, *fixed_component)

        # Run Second Fitting
        print("Fitting all remaining components for segmented IVIM model...")
        results = multithreader(
            self.params.fit_function,
            pixel_args,
            self.params.n_pools if multi_threading else None,
        )
        # Evaluate Results
        self.results.eval_results(results, fixed_component=fixed_component)
        print(
            f"Pixel-wise segmented fitting time: {round(time.time() - start_time, 2)}s"
        )
