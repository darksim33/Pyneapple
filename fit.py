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

    def fitting_pixelwise(self):
        # TODO: add seg number utility for UI purposes
        pixel_args = self.fit_params.get_pixel_args(self.img, self.seg)
        fit_function = self.fit_params.get_fit_function()
        results = fit(fit_function, pixel_args, self.fit_params.n_pools)
        self.fit_results = self.fit_params.eval_pixelwise_fitting_results(
            results, self.seg
        )

    ## NOTE: FOLLOWING CODE NOT IMPLEMENTED IN CODE STRUCTURE YET!
    # TODO: correct implementation of mean? + testing
    def fit_mean_seg(self):
        pixel_args = self.fit_params.get_pixel_args(self.img, self.seg)
        fit_function = self.fit_params.get_fit_function()
        seg_args = np.mean(pixel_args, 3)
        results = fit(fit_function, seg_args, self.fit_params.n_pools)
        self.fit_results = self.fit_params.eval_segmented_fitting_results(
            results, self.seg
        )

def fit(fitfunc, pixel_args, n_pools, debug: bool | None = False):
    # Run Fitting
    # TODO check for max cpu_count()
    debug = True
    if debug:
        results_pixel = []
        for pixel in pixel_args:
            results_pixel.append(fitfunc(pixel[0], pixel[1]))
    else:
        if n_pools != 0:
            with Pool(n_pools) as pool:
                results_pixel = pool.starmap(fitfunc, pixel_args)   
    return results_pixel


# def setup_signalbased_fitting(fit_data: FitData):
#     img = fit_data.img
#     seg = fit_data.seg
#     fit_results = list()
#     for seg_idx in range(1, seg.number_segs + 1, 1):
#         img_seg = seg.get_single_seg_mask(seg_idx)
#         signal = Processing.get_mean_seg_signal(img, img_seg, seg_idx)
#         fit_results.append(fit_segmentation_signal(signal, fit_data, seg_idx))
#     if fit_data.fit_params.model == Model.NNLS:
#         # Create output array for spectrum
#         new_shape = np.array(seg.array.shape)
#         basis = np.exp(
#             -np.kron(
#                 fit_data.fit_params.b_values.T,
#                 fit_data.fit_params.get_bins(),
#             )
#         )
#         new_shape[3] = basis.shape[1]
#         img_results = np.zeros(new_shape)
#         # Sort Entries to array
#         for seg in fit_results:
#             img_results[seg[0]] = seg[1]