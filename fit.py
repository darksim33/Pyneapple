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

    def fit_pixel_wise(self, multi_threading: bool |None = False):
        # TODO: add seg number utility for UI purposes
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)
        fit_function = self.fit_params.get_fit_function()
        results = fit(fit_function, pixel_args, self.fit_params.n_pools, multi_threading = multi_threading)
        self.fit_results = self.fit_params.eval_fitting_results(
            results, self.seg
        )

    def fit_segmentation_wise(self):
        seg_number = self.seg.number_segs
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)
        idx, pixel_args = zip(*list(pixel_args))
        seg_signal = np.mean(pixel_args, axis=0)
        seg_args = (seg_number, seg_signal)
        fit_function = self.fit_params.get_fit_function()
        results = fit(fit_function, seg_args, self.fit_params.n_pools, False)
        self.fit_results = self.fit_params.eval_fitting_results(
            results, self.seg
        )

def fit(fitfunc, fit_args, n_pools, multi_threading: bool | None = True):
    # TODO check for max cpu_count()
    if multi_threading:
        if n_pools != 0:
            with Pool(n_pools) as pool:
                results = pool.starmap(fitfunc, fit_args)           
    else:
        results = []
        for element in fit_args:
            results.append(fitfunc(element[0], element[1]))

    return results