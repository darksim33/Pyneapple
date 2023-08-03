import numpy as np
from multiprocessing import Pool, cpu_count

from utils import Nii, Nii_seg
from fitModel import Model
from fitParameters import *

class FitData:
    def __init__(self, model, img: Nii | None = Nii(), seg: Nii_seg | None = Nii_seg()):
        self.model_name: str | None = model
        self.img = img
        self.seg = seg
        self.fit_results = Results()
        if model == "NNLS":
            self.fit_params = NNLSParams(Model.NNLS)
        elif model == "NNLSreg":
            self.fit_params = NNLSregParams(Model.NNLS)
        elif model == "NNLSregCV":
            self.fit_params = NNLSregCVParams(Model.NNLS_reg_CV)
        elif model == "mono":
            self.fit_params = MonoParams(Model.mono)
        elif model == "mono_T1":
            self.fit_params = MonoT1Params(Model.mono)
        else:
            print("Error: no valid Algorithm")

    def fit_pixelwise(self, debug: bool |None = False):
        # TODO: add seg number utility for UI purposes
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)
        fit_function = self.fit_params.get_fit_function()
        results = fit(fit_function, pixel_args, self.fit_params.n_pools, debug = debug)
        self.fit_results = self.fit_params.eval_fitting_results(
            results, self.seg
        )

    def fit_segmentation_mean(self):
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)
        idx, pixel_args = zip(*list(pixel_args))
        seg_args = np.mean(pixel_args, axis=0)
        fit_function = self.fit_params.get_fit_function()
        # TODO: fit expects tupel of lists, not just one dimensional list
        results = fit(fit_function, seg_args, self.fit_params.n_pools, debug = True)
        self.fit_results = self.fit_params.eval_fitting_results(
            results, self.seg
        )

def fit(fitfunc, fit_args, n_pools, debug: bool | None = False):
    # TODO check for max cpu_count()
    debug = True
    if debug:
        results = []
        for pixel in fit_args:
            results.append(fitfunc(pixel[0], pixel[1]))
    else:
        if n_pools != 0:
            with Pool(n_pools) as pool:
                results = pool.starmap(fitfunc, fit_args)   
    return results