import numpy as np
from multiprocessing import Pool
from pathlib import Path

from src.utils import Nii, NiiSeg
from . import parameters


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
        else:
            self.fit_params = parameters.Parameters(params_json)
            # print("Warning: No valid Fitting Method selected")

    def fit_pixel_wise(self, multi_threading: bool | None = True):
        """Fits every pixel inside the segmentation individually."""

        # TODO: add seg number utility for UI purposes
        pixel_args = self.fit_params.get_element_args(self.img.array, self.seg.array)
        results = fit(
            self.fit_params.fit_function,
            pixel_args,
            self.fit_params.n_pools,
            multi_threading=multi_threading,
        )

        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)

    def fit_segmentation_wise(self):
        """Fits mean signal of segmentation(s), computed of all pixels signals inside."""

        seg_number = list(
            range(self.seg.n_segmentations)
        )  # no information about ROIs location -> single seg only

        # Compute mean signal of seg
        pixel_args = self.fit_params.get_element_args(self.img.array, self.seg.array)
        idx, pixel_args = zip(*list(pixel_args))
        seg_signal = np.mean(pixel_args, axis=0)

        # Create minimal args struct
        seg_args = zip((idx[0],), (seg_signal,))

        seg_result = fit(
            self.fit_params.fit_function, seg_args, self.fit_params.n_pools, False
        )

        # Save result of mean signal for every pixel inside seg
        results = []
        for pixel in idx:
            results.append((pixel, seg_result[0][1]))

        self.fit_results = self.fit_params.eval_fitting_results(results, self.seg)


def fit(fit_function, element_args, n_pools, multi_threading: bool | None = True):
    """Applies correct fitting function, initiates multi-threading if applicable."""

    if multi_threading:  # TODO: check for max cpu_count()
        if n_pools != 0:
            with Pool(n_pools) as pool:
                results = pool.starmap(fit_function, element_args)
    else:
        results = []
        for element in element_args:
            results.append(fit_function(idx=element[0], signal=element[1]))

    return results
