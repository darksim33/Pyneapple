import numpy as np
from scipy.optimize import least_squares, curve_fit, nnls
from scipy import signal
from scipy.sparse import diags

# from typing import Callable
from multiprocessing import Pool, cpu_count
from functools import partial

from utils import Nii, Nii_seg, Processing
from fitting.NNLSregCV import NNLSregCV


class FitModel(object):
    def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int = 200):
        fit, _ = nnls(basis, signal, maxiter=max_iter)
        return idx, fit

    def NNLS_reg_CV(
        idx: int, signal: np.ndarray, basis: np.ndarray, tol: float = 0.0001
    ):
        fit, _, _ = NNLSregCV(basis, signal, tol)
        return idx, fit

    def mono(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        max_iter: int,
    ):
        """Mono exponential Fitting for ADC"""

        # TODO: integrate model_mono directly into curve_fit()?
        def model_mono(b_values: np.ndarray, S0, x0):
            return np.array(S0 * np.exp(-np.kron(b_values, x0)))

        fit, _ = curve_fit(
            model_mono, b_values, signal, x0, bounds=(lb, ub), max_nfev=max_iter
        )
        return idx, fit

    def mono_T1(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        max_iter: int,
        TM: float,
    ):
        """Mono exponential Fitting for ADC and T1"""
        # NOTE does not theme to work at all

        def mono_T1_wrapper(TM: float):
            def mono_T1_model(
                b_values: np.ndarray, S0: float | int, x0: float | int, T1: float | int
            ):
                return np.array(S0 * np.exp(-np.kron(b_values, x0)) * np.exp(-T1 / TM))

            return mono_T1_model

        fit, _ = curve_fit(
            mono_T1_wrapper(TM=TM),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )
        return idx, fit

    def multi_exp(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_components: int,
    ):
        def multi_exp_wrapper(n_components: int):
            def multi_exp_model(b_values: np.ndarray, x0: float | int):
                f = 0
                for i in range(n_components - 2):
                    f = +np.exp(-np.kron(b_values, abs(x0[i]))) * x0[n_components + i]
                return f + np.exp(-np.kron(b_values, abs(x0[n_components - 1]))) * (
                    100 - (np.sum(x0[n_components:]))
                )

            return multi_exp_model

        fit, _ = curve_fit(
            multi_exp_wrapper(n_components=n_components),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit


class FitData:
    def __init__(self, model, img: Nii | None = None, seg: Nii | None = None):
        self.model_name: str | None = model
        self.img = img if img is not None else Nii()
        self.seg = seg if seg is not None else Nii_seg()
        self.fit_results = self.Results()
        if model == "NNLS":
            self.fit_params = NNLSParams(FitModel.NNLS)
        elif model == "NNLSreg":
            self.fit_params = NNLSregParams(FitModel.NNLS)
        elif model == "NNLSregCV":
            self.fit_params = NNLSregCVParams(FitModel.NNLS_reg_CV)
        elif model == "mono":
            self.fit_params = MonoParams(FitModel.mono)
        elif model == "mono_T1":
            self.fit_params = MonoT1Params(FitModel.mono_T1)
        else:
            print("Error no valid Algorithm")

    class Results:
        """
        Class containing estimated diffusion values and fractions

        ...

        Attributes
        ----------

        spectrum :

        d : list
        list of tuples containing pixel coordinates and a np.ndarray holding all the d values
        f : list
        list of tuples containing pixel coordinates and a np.ndarray holding all the f values
        S0 : list
        list of tuples containing pixel coordinates and a np.ndarray holding all the S0 values
        T1 : list
        list of tuples containing pixel coordinates and a np.ndarray holding all the T1 values

        """

        def __init__(self):
            self.spectrum: np.ndarray = np.array([])
            self.d: list | np.ndarray = list() # these should be lists of lists for each parameter
            self.f: list | np.ndarray = list()
            self.S0: list | np.ndarray = list()
            self.T1: list | np.ndarray = list()
            # NOTE paramters lists of tuples containing 

    class Parameters:
        def __init__(
            self,
            model: FitModel | None = None, # Wieso lasssen wir nochmal model = None zu? @TT
            b_values: np.ndarray | None = None,
            n_pools: int | None = 4,  # cpu_count(),
            max_iter: int | None = 250,
        ):
            if not b_values:
                b_values = np.array(
                    [
                        [
                            0,
                            5,
                            10,
                            20,
                            30,
                            40,
                            50,
                            75,
                            100,
                            150,
                            200,
                            250,
                            300,
                            400,
                            525,
                            750,
                        ]
                    ]
                )

            self.model = model
            self.b_values = b_values
            self.max_iter = max_iter
            self.boundaries = self._Boundaries()
            self.variables = self._Variables()
            self.n_pools = n_pools
            self.fit_area = "Pixel"  # Pixel or Segmentation

        # TODO: move/adjust _Boundaries == NNLSParams/MonoParams
        class _Boundaries:
            def __init__(
                self,
                lb: np.ndarray | None = np.array([]),  # lower bound
                ub: np.ndarray | None = np.array([]),  # upper bound
                x0: np.ndarray | None = np.array([]),  # starting values
                # TODO: relocatew n_bins? not a boundary parameter
                n_bins: int | None = 250,
                d_range: np.ndarray
                | None = np.array(
                    [1 * 1e-4, 2 * 1e-1]
                ),  # Lower and Upper Diffusion value for Range
            ):
                # neets fixing based on model maybe change according to model
                # TODO: replace lb and ub by d_range? -> d_range = uniform for all models
                if lb.any():
                    self.lb = lb  # if not lb: np.array([10, 0.0001, 1000])
                if ub.any():
                    self.ub = ub  # if not ub: np.array([1000, 0.01, 2500])
                if x0.any():
                    self.x0 = x0  # if not x0: np.array([50, 0.001, 1750])
                self.n_bins = n_bins
                self.d_range = d_range

        class _Variables:
            def __init__(self, TM: float | None = None):
                self.TM = TM

        def get_bins(self) -> np.ndarray:
            """
            Returns range of Diffusion values for NNLS fitting or plotting
            """
            return np.array(
                np.logspace(
                    np.log10(self.boundaries.d_range[0]),
                    np.log10(self.boundaries.d_range[1]),
                    self.boundaries.n_bins,
                )
            )

        def load_b_values(self, file: str):
            with open(file, "r") as f:
                # find away to decide which one is right
                # self.bvalues = np.array([int(x) for x in f.read().split(" ")])
                self.b_values = np.array([int(x) for x in f.read().split("\n")])

        def get_pixel_args(self, img: Nii, seg: Nii_seg, debug: bool):
            if debug:
                pixel_args = zip(
                    (
                        ((i, j, k), img.array[i, j, k, :])
                        for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                    )
                )
            else:
                pixel_args = zip(
                    (
                        (i, j, k)
                        for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                    ),
                    (
                        img.array[i, j, k, :]
                        for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                    ),
                )
            return pixel_args

    def fitting_pixelwise(self, debug: bool = False):
        # TODO: add seg number utility
        pixel_args = self.fit_params.get_pixel_args(self.img, self.seg, debug)
        fit_function = self.fit_params.get_partial_fit_function()
        results_pixel = fit(fit_function, pixel_args, self.fit_params.n_pools, debug)
        self.fit_results = self.fit_params.eval_pixelwise_fitting_results(
            results_pixel, self.seg
        )


# TODO: update inheritance chain
class NNLSParams(FitData.Parameters):
    def __init__(
        self,
        # TODO: inheritance fix, model & b_values should be inherited without additional initialisation
        model: FitModel | None = FitModel.NNLS,
        max_iter: int | None = 250,
        n_bins: int | None = 250,
        d_range: np.ndarray | None = np.array([1 * 1e-4, 2 * 1e-1]),
        # n_pools: int | None = 4,
    ):
        """
        Basic NNLS Parameter Class
        model: should be of class Model
        """

        if not model:
            super().__init__(FitModel.NNLS)
        else:
            super().__init__(model, max_iter=max_iter)
        self.boundaries.n_bins = n_bins
        self.boundaries.d_range = d_range

    def get_basis(self) -> np.ndarray:
        self._basis = np.exp(
            -np.kron(
                self.b_values.T,
                self.get_bins(),
            )
        )
        return self._basis

    def get_partial_fit_function(self):
        return partial(self.model, basis=self.get_basis())

    def eval_pixelwise_fitting_results(self, results_pixel, seg) -> FitData.Results:
        # Create output array for spectrum
        new_shape = np.array(seg.array.shape)
        new_shape[3] = self.get_basis().shape[1]
        fit_results = FitData.Results()
        fit_results.spectrum = np.zeros(new_shape)
        # Sort entries to array
        for pixel in results_pixel:
            fit_results.spectrum[pixel[0]] = pixel[1]
        # TODO: add d and f and implement find_peaks
        return fit_results


class NNLSregParams(NNLSParams):
    def __init__(
        self,
        model: FitModel | None = FitModel.NNLS,
        reg_order: int | None = 2,
        mu: float | None = 0.01,
    ):
        super().__init__(
            model=model,
            max_iter=100000,
        )
        self.reg_order = reg_order
        self.mu = mu

    def get_basis(self) -> np.ndarray:
        basis = super().get_basis()
        n_bins = self.boundaries.n_bins

        if self.reg_order == 0:
            # no weighting
            reg = diags([1], [0], shape=(n_bins, n_bins)).toarray()
        elif self.reg_order == 1:
            # weighting with the predecessor
            reg = diags([-1, 1], [0, 1], shape=(n_bins, n_bins)).toarray()
        elif self.reg_order == 2:
            # weighting of the nearest neighbours
            reg = diags([1, -2, 1], [-1, 0, 1], shape=(n_bins, n_bins)).toarray()
        elif self.reg_order == 3:
            # weighting of the first and second nearest neighbours
            reg = diags(
                [1, 2, -6, 2, 1], [-2, -1, 0, 1, 2], shape=(n_bins, n_bins)
            ).toarray()

        # append reg to create regularised NNLS basis
        return np.concatenate((basis, reg * self.mu))

    def get_pixel_args(self, img: Nii, seg: Nii_seg, debug: bool):
        # enhance image array for regularisation
        reg = np.zeros((np.append(np.array(img.array.shape[0:3]), 250)))
        img_reg = np.concatenate((img.array, reg), axis=3)

        # TODO: understand code @TT
        if debug:
            pixel_args = zip(
                (
                    ((i, j, k), img_reg[i, j, k, :])
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                )
            )
        else:
            pixel_args = zip(
                (
                    (i, j, k)
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                ),
                (
                    img_reg[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                ),
            )

        return pixel_args


class NNLSregCVParams(NNLSParams):
    def __init__(
        self, model: FitModel | None = FitModel.NNLS_reg_CV, tol: float | None = 0.0001
    ):
        super().__init__(model=model)
        self.tol = tol


class MonoParams(FitData.Parameters):
    def __init__(
        self,
        model: FitModel | None = FitModel.mono,
        x0: np.ndarray | None = np.array([50, 0.001]),
        lb: np.ndarray | None = np.array([10, 0.0001]),
        ub: np.ndarray | None = np.array([1000, 0.01]),
        max_iter: int | None = 600,
    ):
        super().__init__(model=model, max_iter=max_iter)
        self.boundaries.x0 = x0
        self.boundaries.lb = lb
        self.boundaries.ub = ub

    def get_basis(self):
        # BUG Bvlaues are passed in the wrong shape
        return np.squeeze(self.b_values)

    def get_partial_fit_function(self):
        return partial(
            self.model,
            b_values=self.get_basis(),
            x0=self.boundaries.x0,
            lb=self.boundaries.lb,
            ub=self.boundaries.ub,
            max_iter=self.max_iter,
        )

    def eval_pixelwise_fitting_results(self, results_pixel, seg) -> FitData.Results:
        # prepare arrays 
        fit_results = FitData.Results()
        for pixel in results_pixel:
            fit_results.S0.append((pixel[0],[pixel[1][0]]))
            fit_results.d.append((pixel[0],[pixel[1][1]]))
            fit_results.f.append((pixel[0],np.ones(1)))
        
        # NOTE for T1 just all super and then load results again for aditional T1 values

        fit_results = self.set_spectrum_from_variables(fit_results, seg)

        return fit_results

    def set_spectrum_from_variables(self, fit_results: FitData.Results, seg: Nii_seg):
        # adjust D values acording to bins/dvalues
        d_values = self.get_bins()
        d_new = np.zeros(len(fit_results.d[1]))

        new_shape = np.array(seg.array.shape)
        new_shape[3] = self.boundaries.n_bins
        spectrum = np.zeros(new_shape)

        for d_pixel, f_pixel in zip(fit_results.d, fit_results.f):
            temp_spec = np.zeros(self.boundaries.n_bins)
            for idx, (D, F) in enumerate(zip(d_pixel[1], f_pixel[1])):
                index = np.unravel_index(
                    np.argmin(abs(d_values - D), axis=None),
                    d_values.shape,
                )[0].astype(int)
                d_new[idx] = d_values[index]
                temp_spec = temp_spec + F * signal.unit_impulse(
                    self.boundaries.n_bins, index
                )
            spectrum[d_pixel[0]] = temp_spec
        fit_results.spectrum = spectrum
        return fit_results


class MonoT1Params(MonoParams):
    def __init__(
        self,
        model: FitModel | None = FitModel.mono_T1,
        x0: np.ndarray | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
        TM: float | None = None,
        max_iter: int | None = 600,
    ):
        super().__init__(model=model, max_iter=max_iter)
        # TODO: super needed?
        # Andere boundaries als mono? @TT
        if model == FitModel.mono_T1:
            self.boundaries.x0 = x0 if x0 is not None else np.array([50, 0.001, 1750])
            self.boundaries.lb = lb if lb is not None else np.array([10, 0.0001, 1000])
            self.boundaries.ub = ub if ub is not None else np.array([1000, 0.01, 2500])
            self.variables.TM = TM if TM is not None else 20.0

    # TODO: same as in MonoT1 and NNLS -> inherit functions?
    def get_partial_fit_function(self):
        return partial(
            self.model,
            b_values=self.get_basis(),
            x0=self.boundaries.x0,
            lb=self.boundaries.lb,
            ub=self.boundaries.ub,
            TM=self.variables.TM,
            max_iter=self.max_iter,
        )
    
    def eval_pixelwise_fitting_results(self, results_pixel, seg) -> FitData.Results:
        fit_results = super().eval_pixelwise_fitting_results(results_pixel, seg)
        for pixel in results_pixel:
            fit_results.T1.append((pixel[0],[pixel[1][2]]))
        return fit_results

# def setup_signalbased_fitting(fit_data: FitData):
#     img = fit_data.img
#     seg = fit_data.seg
#     fit_results = list()
#     for seg_idx in range(1, seg.number_segs + 1, 1):
#         img_seg = seg.get_single_seg_mask(seg_idx)
#         signal = Processing.get_mean_seg_signal(img, img_seg, seg_idx)
#         fit_results.append(fit_segmentation_signal(signal, fit_data, seg_idx))
#     if fit_data.fit_params.model == FitModel.NNLS:
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


# def fit_segmentation_signal(
#     signal: np.ndarray, fit_params: FitData.Parameters, seg_idx: int
# ):
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


# TODO: MOVE!
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
