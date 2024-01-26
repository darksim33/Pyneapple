import numpy as np
from multiprocessing import Pool
from pathlib import Path

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
        # TODO: add seg number utility for UI purposes
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)

        results = multithreader(
            self.fit_params.fit_function,
            pixel_args,
            self.fit_params.n_pools if multi_threading else None,
        )

        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)

    def fit_segmentation_wise(self):
        # TODO: implement counting of segmentations via range?
        seg_number = list([self.seg.n_segmentations])
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)
        idx, pixel_args = zip(*list(pixel_args))
        seg_signal = np.mean(pixel_args, axis=0)
        seg_args = (seg_number, seg_signal)
        results = multithreader(
            self.fit_params.fit_function,
            seg_args,
            n_pools=None,  # self.fit_params.n_pools,
        )
        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)

    def fit_ideal(self, multi_threading: bool = False, debug: bool = False):
        if not self.model_name == "IDEAL":
            raise AttributeError("Wrong model name!")
        print(f"The initial image size is {self.img.array.shape[0:4]}.")
        fit_params = fit_ideal(
            self.img, self.seg, self.fit_params, 0, multi_threading, debug
        )
        self.fit_results = self.fit_params.eval_fitting_results(fit_params, self.seg)
