import numpy as np
from functools import partial
from multiprocessing import Pool
# from fitData import FitData

def fit(fitfunc, pixel_args, n_pools, debug: bool | None = False):
    # Run Fitting
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


## NOTE: FOLLOWING CODE NOT IMPLEMENTED IN CODE STRUCTURE YET!
def fit_pixelwise(fit_data, debug: bool = False):
    # TODO: add seg number utility
    pixel_args = fit_data.fit_params.get_pixel_args(fit_data.img, fit_data.seg, debug)
    fit_function = fit_data.fit_params.get_fit_function()
    results = fit(fit_function, pixel_args, fit_data.fit_params.n_pools, debug)
    fit_data.fit_results = fit_data.fit_params.eval_pixelwise_fitting_results(
        results, fit_data.seg
    )


# correct implementation of mean?
# TODO: implement eval_segmented_fitting_results
def fit_mean_seg(self):
    pixel_args = self.fit_params.get_pixel_args(self.img, self.seg)
    fit_function = self.fit_params.get_fit_function()
    seg_args = np.mean(pixel_args, 3)
    results = fit(fit_function, seg_args, self.fit_params.n_pools)
    self.fit_results = self.fit_params.eval_segmented_fitting_results(
        results, self.seg
    )


# def fit_segmentation_signal(signal: np.ndarray, fit_params: FitData.Parameters, seg_idx: int):
#     if fit_params.model == FitModel.NNLS:
#         basis = np.exp(
#             -np.kron(
#                 fit_params.b_values.T,
#                 fit_params.get_bins(),
#             )
#         )
#         fit_function = partial(fit_params.model, basis=basis)
#     elif fit_params.model == FitModel.mono:
#         print("test")
#     return fit_function(seg_idx, signal)
#
#
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