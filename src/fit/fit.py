import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from src.utils import Nii, NiiSeg
from . import parameters
from src.multithreading import multithreader
from src.fit.ideal import fit_ideal


class FitData:
    """
    Fitting class for (multi-threaded) pixel- and segmentation-wise fitting.

    Attributes
    ----------
    model : str
    img : Nii
    seg : NiiSeg

    Methods
    --------
    fit_pixel_wise(multi_threading: bool | None = True)
        Fits every pixel inside the segmentation individually. Multi-threading possible to boost performance.
    fit_segmentation_wise()
        Fits mean signal of segmentation(s).
    """

    def __init__(
        self,
        model: str | None = None,
        params_json: str | Path | None = None,
        img: Nii | None = Nii(),
        seg: NiiSeg | None = NiiSeg(),
    ):
        self.model_name = model
        self.img = img
        self.seg = seg
        self.fit_results = parameters.Results()
        if model == "NNLS":
            self.fit_params = parameters.NNLSParams(params_json)
        elif model == "NNLSreg":
            self.fit_params = parameters.NNLSregParams(params_json)
        elif model == "NNLSregCV":
            self.fit_params = parameters.NNLSregCVParams(params_json)
        elif model == "IVIM":
            self.fit_params = parameters.IVIMParams(params_json)
        elif model == "IDEAL":
            self.fit_params = parameters.IDEALParams(params_json)
        else:
            self.fit_params = parameters.Parameters(params_json)
            # print("Warning: No valid Fitting Method selected")

    def fit_pixel_wise(self, multi_threading: bool | None = True):
        """Fits every pixel inside the segmentation individually."""
        start_time = time.time()
        # TODO: add seg number utility for UI purposes
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)

        results = multithreader(
            self.fit_params.fit_function,
            pixel_args,
            self.fit_params.n_pools if multi_threading else None,
        )

        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)
        print(f"Pixel wise time:{round(time.time() - start_time, 2)}s")

    def fit_segmentation_wise(self, **kwargs):
        """Fits mean signal of segmentation(s), computed of all pixels signals."""
        start_time = time.time()
        results = list()
        for seg_number in self.seg.seg_numbers.astype(int):
            # get mean pixel signal
            seg_signal = self.seg.get_mean_signal(self.img.array, seg_number)
            seg_args = zip([[seg_number]], [seg_signal])
            # fit mean signal
            seg_results = multithreader(
                self.fit_params.fit_function,
                seg_args,
                n_pools=None,  # self.fit_params.n_pools,
            )

            # Save result of mean signal for every pixel of each seg
            for pixel in self.seg.get_seg_indices(seg_number):
                results.append((pixel, seg_results[0][1]))

        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)
        print(f"{round(time.time() - start_time, 2)}s")

    def fit_ideal(self, multi_threading: bool = False, debug: bool = False):
        """IDEAL Fitting Interface."""
        start_time = time.time()
        if not self.model_name == "IDEAL":
            raise AttributeError("Wrong model name!")
        print(f"The initial image size is {self.img.array.shape[0:4]}.")
        fit_results = fit_ideal(
            self.img, self.seg, self.fit_params, multi_threading, debug
        )
        self.fit_results = self.fit_params.eval_fitting_results(fit_results, self.seg)
        print(f"IDEAL time:{round(time.time() - start_time, 2)}s")
