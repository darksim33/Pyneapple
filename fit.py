import numpy as np
from functools import partial
from multiprocessing import Pool


def fit(fitfunc, pixel_args, n_pools, debug: bool | None = False):
    # Run Fitting
    if debug:
        results_pixel = []
        for pixel in pixel_args:
            results_pixel.append(fitfunc(pixel[0][0], pixel[0][1]))
    else:
        if n_pools != 0:
            with Pool(n_pools) as pool:
                results_pixel = pool.starmap(fitfunc, pixel_args)
    return results_pixel


def fit_pixelwise(self, debug: bool = False):
    # TODO: add seg number utility
    pixel_args = self.fit_params.get_pixel_args(self.img, self.seg, debug)
    fit_function = self.fit_params.get_fit_function()
    results = fit(fit_function, pixel_args, self.fit_params.n_pools, debug)
    self.fit_results = self.fit_params.eval_pixelwise_fitting_results(
        results, self.seg
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


# def fit_seg_mean(signal: np.ndarray, fit_params: FitData.Parameters, seg_idx: int):
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