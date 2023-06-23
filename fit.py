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

    def NNLS_reg(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int = 200):
        fit, _ = nnls(basis, signal, maxiter=max_iter)
        return idx, fit

    def mono(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        """Mono exponential Fitting for ADC"""

        # TODO: integrate model_mono directly into curve_fit()?
        def model_mono(b_values: np.ndarray, S0, x0):
            return np.array(S0 * np.exp(-np.kron(b_values, x0)))

        fit, _ = curve_fit(
            model_mono,
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit

    def mono_T1(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        TM: int,
    ):
        """Mono exponential Fitting for ADC and T1"""

        def mono_T1_wrapper(TM: int):
            def mono_T1_model(
                b_values: np.ndarray, S0: float | int, x0: float | int, T1: float | int
            ):
                return np.array(S0 * np.exp(-np.kron(b_values, x0)) * np.exp(-T1 / TM))

            return mono_t1_model

        fit, _ = curve_fit(
            mono_T1_wrapper(TM=TM),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
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
            self.fit_params = NNLSregParams(FitModel.NNLS_reg)
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
        """

        def __init__(self):
            self.spectrum: np.ndarray = np.array([])
            self.d: np.ndarray = np.array([])
            self.f: np.ndarray = np.array([])
            self.S0: np.ndarray = np.array([])
            self.T1: np.ndarray = np.array([])

    def set_spectrum_from_variables(self):
        # adjust D values acording to bins/dvalues
        d_values = self.fit_params.get_d_values()
        d_new = np.zeros(len(self.fit_results.d[1]))

        new_shape = np.array(self.seg.array.shape)
        new_shape[3] = self.fit_params.boundaries.n_bins
        spectrum = np.zeros(new_shape)

        for d_pixel, f_pixel in zip(self.fit_results.d, self.fit_results.f):
            temp_spec = np.zeros(self.fit_params.boundaries.n_bins)
            for idx, D in enumerate(d_pixel[1]):
                index = np.unravel_index(
                    np.argmin(abs(d_values - D), axis=None),
                    d_values.shape,
                )[0].astype(int)
                d_new[idx] = d_values[index]
                temp_spec = temp_spec + f_pixel[1][idx] * signal.unit_impulse(
                    self.fit_params.boundaries.n_bins, index
                )
            spectrum[d_pixel[0]] = temp_spec
        self.fit_results.spectrum = spectrum

    class Parameters:
        def __init__(
            self,
            model: FitModel | None = None,
            b_values: np.ndarray | None = None,
            nPools: int | None = 4,  # cpu_count(),
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
            self.boundaries = self._Boundaries()
            self.variables = self._Variables()
            self.nPools = nPools

        # TODO: move/adjust _Boundaries == NNLSParams/MonoParams
        class _Boundaries:
            def __init__(
                self,
                lb: np.ndarray | None = np.array([]),  # lower bound
                ub: np.ndarray | None = np.array([]),  # upper bound
                x0: np.ndarray | None = np.array([]),  # starting values
                # TODO: relocatew n_bins? not a boundary parameter
                n_bins: int | None = 250,  # Number of functions for NNLS
                d_range: np.ndarray
                | None = np.array(
                    [1 * 1e-4, 2 * 1e-1]
                ),  # Lower and Upper Diffusion value for Range
                # TM: np.ndarray | None = None,  # mixing time
            ):
                # neets fixing based on model maybe change according to model
                if lb.any():
                    self.lb = lb  # if lb is not None else np.array([10, 0.0001, 1000])
                if ub.any():
                    self.ub = ub  # if ub is not None else np.array([1000, 0.01, 2500])
                if x0.any():
                    self.x0 = x0  # if x0 is not None else np.array([50, 0.001, 1750])
                # bins and d_range are always used to create diffusion distribution
                self.n_bins = n_bins
                self.d_range = d_range

        class _Variables:
            def __init__(self, TM: float | None = None):
                self.TM = TM

        def get_d_values(self) -> np.ndarray:
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
        results_pixel = fit(fit_function, pixel_args, self.fit_params.nPools, debug)
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
        # nPools: int | None = 4,
    ):
        """
        Basic NNLS Parameter Class
        model: should be of class Model
        """
        # why not: "if not b_values np.array(...)" ?

        if not model:
            super().__init__(FitModel.NNLSs)
        else:
            super().__init__(model)
        self.boundaries.n_bins = n_bins
        self.boundaries.d_range = d_range
        self.max_iter = max_iter

    def get_basis(self) -> np.ndarray:
        self._basis = np.exp(
            -np.kron(
                self.b_values.T,
                self.get_d_values(),
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
        model: FitModel | None = FitModel.NNLS_reg,
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
    ):
        super().__init__(model=model)
        self.boundaries.x0 = x0
        self.boundaries.lb = lb
        self.boundaries.ub = ub

    def get_basis(self):
        return self.b_values

    # def evaluateFit(self,fit_results,pixe):
    #     if self.model == fitModels.mono_fit:
    #     elif self.model == fitModels.mono_T1_fit:
    #         fit_results.S0 = [None] * len(list(results_pixel))
    #         fit_results.d = [None] * len(list(results_pixel))
    #         fit_results.T1 = [None] * len(list(results_pixel))
    #         fit_results.f = [None] * len(list(results_pixel))
    #         for idx, pixel in enumerate(results_pixel):
    #             fit_results.S0[idx] = (pixel[0], np.array([pixel[1][0]]))
    #             fit_results.d[idx] = (pixel[0], np.array([pixel[1][1]]))
    #             fit_results.T1[idx] = (pixel[0], np.array([pixel[1][2]]))
    #             fit_results.f[idx] = (pixel[0], np.array([1]))
    #         FitData.fit_results = fit_results
    #         FitData.set_spectrum_from_variables()


class MonoT1Params(MonoParams):
    def __init__(
        self,
        model: FitModel | None = FitModel.mono_T1,
        x0: np.ndarray | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
    ):
        super().__init__(model=model)
        if model == FitModel.mono_T1:
            self.boundaries.x0 = x0 if x0 is not None else np.array([50, 0.001, 1750])
            self.boundaries.lb = lb if lb is not None else np.array([10, 0.0001, 1000])
            self.boundaries.ub = ub if ub is not None else np.array([1000, 0.01, 2500])


def setup_pixelwise_fitting(fit_data, debug: bool | None = False) -> Nii:
    # prepare Workers
    img = fit_data.img
    seg = fit_data.seg
    fit_params = fit_data.fit_params

    if debug:
        pixel_args = zip(
            (
                ((i, j, k), img.array[i, j, k, :])
                for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
            ),
        )
    else:
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))),
            (
                img.array[i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
            ),
        )
    if fit_params.model == FitModel.NNLS or fit_params.model == FitModel.NNLS_reg_CV:
        # Prepare basis for NNLS from b_values and
        basis = np.exp(
            -np.kron(
                fit_params.b_values.T,
                fit_params.get_d_values(),
            )
        )
        fitfunc = partial(fit_params.model, basis=basis)
    elif fit_params.model == FitModel.mono:
        basis = fit_params.b_values
        fitfunc = partial(
            fit_params.model,
            b_values=basis,
            x0=fit_params.Bounds.x0,
            lb=fit_params.Bounds.lb,
            ub=fit_params.Bounds.ub,
        )
    elif fit_params.model == FitModel.mono_T1:
        basis = fit_params.b_values
        fitfunc = partial(
            fit_params.model,
            b_values=basis,
            x0=fit_params.Bounds.x0,
            lb=fit_params.Bounds.lb,
            ub=fit_params.Bounds.ub,
            TM=fit_params.variables.TM,
        )

    results_pixel = fit(fitfunc, pixel_args, fit_params.nPools, debug)

    # Sort Results
    fit_results = fit_data.fit_results
    if fit_params.model == FitModel.NNLS or fit_params.model == FitModel.NNLS_reg_CV:
        # Create output array for spectrum
        new_shape = np.array(seg.array.shape)
        new_shape[3] = basis.shape[1]
        fit_results.spectrum = np.zeros(new_shape)
        # Sort Entries to array
        for pixel in results_pixel:
            fit_results.spectrum[pixel[0]] = pixel[1]
        # TODO: add d and f
    elif fit_params.model == FitModel.mono:
        fit_results.S0 = [None] * len(list(results_pixel))
        fit_results.d = [None] * len(list(results_pixel))
        fit_results.f = [None] * len(list(results_pixel))
        for idx, pixel in enumerate(results_pixel):
            fit_results.S0[idx] = (pixel[0], np.array([pixel[1][0]]))
            fit_results.d[idx] = (pixel[0], np.array([pixel[1][1]]))
            fit_results.f[idx] = (pixel[0], np.array([1]))
        fit_data.fit_results = fit_results
        fit_data.set_spectrum_from_variables()
    elif fit_params.model == FitModel.mono_T1:
        fit_results.S0 = [None] * len(list(results_pixel))
        fit_results.d = [None] * len(list(results_pixel))
        fit_results.T1 = [None] * len(list(results_pixel))
        fit_results.f = [None] * len(list(results_pixel))
        for idx, pixel in enumerate(results_pixel):
            fit_results.S0[idx] = (pixel[0], np.array([pixel[1][0]]))
            fit_results.d[idx] = (pixel[0], np.array([pixel[1][1]]))
            fit_results.T1[idx] = (pixel[0], np.array([pixel[1][2]]))
            fit_results.f[idx] = (pixel[0], np.array([1]))
        fit_data.fit_results = fit_results
        fit_data.set_spectrum_from_variables()
    # TODO: remove?!
    return Nii().from_array(fit_data.fit_results.spectrum)


def setup_signalbased_fitting(fit_data: FitData):
    img = fit_data.img
    seg = fit_data.seg
    fit_results = list()
    for seg_idx in range(1, seg.number_segs + 1, 1):
        img_seg = seg.get_single_seg_mask(seg_idx)
        signal = Processing.get_mean_seg_signal(img, img_seg, seg_idx)
        fit_results.append(fit_segmentation_signal(signal, fit_data, seg_idx))
    if fit_data.fit_params.model == FitModel.NNLS:
        # Create output array for spectrum
        new_shape = np.array(seg.array.shape)
        basis = np.exp(
            -np.kron(
                fit_data.fit_params.b_values.T,
                fit_data.fit_params.get_d_values(),
            )
        )
        new_shape[3] = basis.shape[1]
        img_results = np.zeros(new_shape)
        # Sort Entries to array
        for seg in fit_results:
            img_results[seg[0]] = seg[1]


def fit_segmentation_signal(
    signal: np.ndarray, fit_params: FitData.Parameters, seg_idx: int
):
    if fit_params.model == FitModel.NNLS:
        basis = np.exp(
            -np.kron(
                fit_params.b_values.T,
                fit_params.get_d_values(),
            )
        )
        fit_function = partial(fit_params.model, basis=basis)
    elif fit_params.model == FitModel.mono:
        print("test")
    return fit_function(seg_idx, signal)


# TODO: MOVE!
def fit(fitfunc, pixel_args, nPools, debug: bool | None = False):
    # Run Fitting
    if debug:
        results_pixel = []
        for pixel in pixel_args:
            results_pixel.append(fitfunc(pixel[0][0], pixel[0][1]))
    else:
        if nPools != 0:
            with Pool(nPools) as pool:
                results_pixel = pool.starmap(fitfunc, pixel_args)
    return results_pixel
