import numpy as np
from multiprocessing import Pool, cpu_count

from src.utils import Nii, NiiSeg
from .model import Model
from . import parameters


class FitData:
    def __init__(
        self,
        model: str | None = None,
        img: Nii | None = Nii(),
        seg: NiiSeg | None = NiiSeg(),
    ):
        self.model_name = model
        self.img = img
        self.seg = seg
        self.fit_results = parameters.Results()
        if model == "NNLS":
            self.fit_params = parameters.NNLSParams(Model.NNLS)
        elif model == "NNLSreg":
            self.fit_params = parameters.NNLSRegParams(Model.NNLS)
        elif model == "NNLSregCV":
            self.fit_params = parameters.NNLSregCVParams(Model.NNLS_reg_CV)
        elif model == "mono":
            self.fit_params = parameters.MonoParams(Model.mono)
        elif model == "mono_T1":
            self.fit_params = parameters.MonoT1Params(Model.mono)
        elif model == "multiExp":
            self.fit_params = parameters.MultiTest()
        else:
            self.fit_params = parameters.Parameters()
            print("Warning: No valid Fitting Method selected")

    def fit_pixel_wise(self, multi_threading: bool | None = True):
        # TODO: add seg number utility for UI purposes
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)
        fit_function = self.fit_params.get_fit_function()
        results = fit(
            fit_function,
            pixel_args,
            self.fit_params.n_pools,
            multi_threading=multi_threading,
        )
        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)

    def fit_segmentation_wise(self):
        # TODO: implement counting of segmentations via range?
        seg_number = list([self.seg.number_segs])
        pixel_args = self.fit_params.get_pixel_args(self.img.array, self.seg.array)
        idx, pixel_args = zip(*list(pixel_args))
        seg_signal = np.mean(pixel_args, axis=0)
        seg_args = (seg_number, seg_signal)
        fit_function = self.fit_params.get_fit_function()
        results = fit(fit_function, seg_args, self.fit_params.n_pools, False)
        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)


def fit(fit_func, element_args, n_pools, multi_threading: bool | None = True):
    # TODO check for max cpu_count()
    if multi_threading:
        if n_pools != 0:
            with Pool(n_pools) as pool:
                results = pool.starmap(fit_func, element_args)
    else:
        results = []
        for element in element_args:
            results.append(fit_func(element[0], element[1]))

    return results
