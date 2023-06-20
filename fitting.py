import numpy as np
from scipy.optimize import least_squares, curve_fit, nnls
from scipy import signal

# from typing import Callable
from multiprocessing import Pool, cpu_count
from functools import partial

from utils import Nii, Nii_seg, Processing
from fitting.NNLSregCV import NNLSregCV


class fitData:
    def __init__(self, model_name, img: Nii | None = None, mask: Nii | None = None):
        self.model_name: str | None = model_name
        self.img = img if img is not None else Nii()
        self.mask = mask if mask is not None else Nii_seg()
        self.fit_params = self.fitParameters(fit_model=None)
        self.fit_results = self._fitResults()

    class _fitResults:
        """
        Class containing Diffusion values and Fractions
        """

        def __init__(self):
            self.spectrum: np.ndarray
            self.Ds = []
            self.Fs = []
            self.S0s = []
            self.T1s = []

    def set_spectrum_from_variables(self):
        # adjust D values acording to bins/dvalues
        DValues = self.fit_params.get_DValues()
        DsNew = np.zeros(len(self.fit_results.Ds[1]))

        new_shape = np.array(self.mask.array.shape)
        new_shape[3] = self.fit_params.Bounds.nbins
        spectrum = np.zeros(new_shape)

        for pixel_Ds, pixel_Fs in zip(self.fit_results.Ds, self.fit_results.Fs):
            temp_spec = np.zeros(self.fit_params.Bounds.nbins)
            for idx, D in enumerate(pixel_Ds[1]):
                index = np.unravel_index(
                    np.argmin(abs(DValues - D), axis=None),
                    DValues.shape,
                )[0].astype(int)
                DsNew[idx] = DValues[index]
                temp_spec = temp_spec + pixel_Fs[1][idx] * signal.unit_impulse(
                    self.fit_params.Bounds.nbins, index
                )
            spectrum[pixel_Ds[0]] = temp_spec
        self.fit_results.spectrum = spectrum

    class fitParameters:
        def __init__(
            self,
            fit_model,  #: Callable | None = None,
            bValues: np.ndarray | None = np.array([]),
            nPools: int = 4,  # cpu_count(),
        ):
            self.fit_model = fit_model
            self.bValues = bValues
            self.Bounds = self._fitBoundries()
            self.variables = self._variables()
            self.nPools = nPools

        class _fitBoundries:
            def __init__(
                self,
                lb: np.ndarray | None = None,  # lower bound
                ub: np.ndarray | None = None,  # upper bound
                x0: np.ndarray | None = None,  # starting values
                nbins: int | None = 250,  # Number of functions for NNLS
                DiffBounds: np.ndarray
                | None = np.array(
                    [1 * 1e-4, 2 * 1e-1]
                ),  # Lower and Upper Diffusion value for Range
                # TM: np.ndarray | None = None,  # mixing time
            ):
                # neets fixing based on model maybe change according to model
                self.lb = lb  # if lb is not None else np.array([10, 0.0001, 1000])
                self.ub = ub  # if ub is not None else np.array([1000, 0.01, 2500])
                self.x0 = x0  # if x0 is not None else np.array([50, 0.001, 1750])
                self.nbins = nbins
                self.DiffBounds = DiffBounds

        class _variables:
            def __init__(self, TM: float | None = None):
                self.TM = TM

        def get_DValues(self) -> np.ndarray:
            """
            Returns range of Diffusion values for NNLS fitting or plotting
            """
            return np.array(
                np.logspace(
                    np.log10(self.Bounds.DiffBounds[0]),
                    np.log10(self.Bounds.DiffBounds[1]),
                    self.Bounds.nbins,
                )
            )

        def load_bvals(self, file: str):
            with open(file, "r") as f:
                # find away to decide which one is right
                # self.bvalues = np.array([int(x) for x in f.read().split(" ")])
                self.bValues = np.array([int(x) for x in f.read().split("\n")])


class NNLSParams(fitData.fitParameters):
    def __init__(
        self,
        model: str | None = "NNLS",
        bValues: np.ndarray | None = None,
        nbins: int | None = 250,
        DiffBounds: np.ndarray | None = np.array([1 * 1e-4, 2 * 1e-1]),
        nPools: int | None = None,
    ):
        bValues = (
            bValues
            if bValues is not None
            else np.array(
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
        )
        if model == "NNLS":
            super().__init__(FitModels.NNLS, bValues, nPools=nPools)
        elif model == "NNLSreg":
            super().__init__(FitModels.NNLSreg, bValues, nPools=nPools)
        self.Bounds.nbins = nbins
        self.Bounds.DiffBounds = DiffBounds

    @property
    def max_iters(self):
        return self._max_iters

    @max_iters.setter
    def max_iters(self, max_iterations):
        self._max_iters = max_iterations

    def get_basis(self):
        self._basis = np.exp(
            -np.kron(
                self.bValues.T,
                self.get_DValues(),
            )
        )
        return self._basis

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


class NNLSregParams(NNLSParams):
    def __init__(
        self,
        # model: str | None,
        bValues: np.ndarray | None,
        nbins: int | None,
        DiffBounds: np.ndarray | None,
        nPools: int | None = None,
        reg_order: int | None = 2,
        mu: float | None = 0.01,
    ):
        super().__init__(
            model="NNLSreg",
            bValues=bValues,
            nbins=nbins,
            DiffBounds=DiffBounds,
            nPools=nPools,
        )
        self._reg_order = reg_order
        self._mu = mu

    def get_basis(self):
        basis = np.exp(
            -np.kron(
                self.bValues.T,
                self.get_DValues(),
            )
        )
        n_data = len(self.bValues)
        n_bins = len(self.Bounds.nbins)

        # create new basis and signal
        basis_new = np.zeros([n_data + n_bins, n_bins])
        # add current set to new one
        basis_new[0:n_data, 0:n_bins] = basis

        for i in range(n_bins, (n_bins + n_data), 1):
            # idx_data is iterator for the datapoints
            # since the new basis is already filled with the basis set it only needs to iterate beyond that
            for j in range(n_bins):
                # idx_bin is the iterator for the bins
                basis_new[i, j] = 0
                if self._reg_order == 0:
                    # no weighting
                    if i - n_data == j:
                        basis_new[i, j] = 1.0 * self._mu
                elif self._reg_order == 1:
                    # weighting with the predecessor
                    if i - n_data == j:
                        basis_new[i, j] = -1.0 * self._mu
                    elif i - n_data == j + 1:
                        basis_new[i, j] = 1.0 * self._mu
                elif self._reg_order == 2:
                    # weighting of the nearest neighbours
                    if i - n_data == j - 1:
                        basis_new[i, j] = 1.0 * self._mu
                    elif i - n_data == j:
                        basis_new[i, j] = -2.0 * self._mu
                    elif i - n_data == j + 1:
                        basis_new[i, j] = 1.0 * self._mu
                elif self._reg_order == 3:
                    # weighting of the first and second nearest neighbours
                    if i - n_data == j - 2:
                        basis_new[i, j] = 1.0 * self._mu
                    elif i - n_data == j - 1:
                        basis_new[i, j] = 2.0 * self._mu
                    elif i - n_data == j:
                        basis_new[i, j] = -6.0 * self._mu
                    elif i - n_data == j + 1:
                        basis_new[i, j] = 2.0 * self._mu
                    elif i - n_data == j + 2:
                        basis_new[i, j] = 1.0 * self._mu

        return basis_new

    def get_pixel_args(self, img: Nii, seg: Nii_seg, debug: bool):
        # enhance image array for regularisation
        img_new = np.zeros(
            [
                img.array.shape[0],
                img.array.shape[1],
                img.array.shape[2],
                (img.array.shape[3] + self.Bounds.nbins),
            ]
        )
        img_new[
            0 : img.array.shape[0],
            0 : img.array.shape[1],
            0 : img.array.shape[2],
            0 : img.array.shape[3],
        ] = img.array
        if debug:
            pixel_args = zip(
                (
                    ((i, j, k), img_new[i, j, k, :])
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
                    img_new.array[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                ),
            )
        return pixel_args


class MonoParams(fitData.fitParameters):
    def __init__(
        self,
        model: str | None = None,
        bValues: np.ndarray | None = np.array([]),
        nPools: int = 4,
        x0: np.ndarray | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
    ):
        if model == "mono":
            super().__init__(
                fit_model=FitModels.monoFit, bValues=bValues, nPools=nPools
            )
            self.Bounds.x0 = x0 if x0 is not None else np.array([50, 0.001])
            self.Bounds.lb = lb if lb is not None else np.array([10, 0.0001])
            self.Bounds.ub = ub if ub is not None else np.array([1000, 0.01])
        elif model == "mono_t1":
            super().__init__(
                fit_model=FitModels.mono_t1Fit, bValues=bValues, nPools=nPools
            )
            self.Bounds.x0 = x0 if x0 is not None else np.array([50, 0.001, 1750])
            self.Bounds.lb = lb if lb is not None else np.array([10, 0.0001, 1000])
            self.Bounds.ub = ub if ub is not None else np.array([1000, 0.01, 2500])
        else:
            print("ERROR")

    def get_basis(self):
        return self.bValues

    # def evaluateFit(self,fit_results,pixe):
    #     if self.fit_model == fitModels.monoFit:
    #     elif self.fit_model == fitModels.mono_t1Fit:
    #         fit_results.S0s = [None] * len(list(pixel_results))
    #         fit_results.Ds = [None] * len(list(pixel_results))
    #         fit_results.T1s = [None] * len(list(pixel_results))
    #         fit_results.Fs = [None] * len(list(pixel_results))
    #         for idx, pixel in enumerate(pixel_results):
    #             fit_results.S0s[idx] = (pixel[0], np.array([pixel[1][0]]))
    #             fit_results.Ds[idx] = (pixel[0], np.array([pixel[1][1]]))
    #             fit_results.T1s[idx] = (pixel[0], np.array([pixel[1][2]]))
    #             fit_results.Fs[idx] = (pixel[0], np.array([1]))
    #         fitData.fit_results = fit_results
    #         fitData.set_spectrum_from_variables()


class FitModels(object):
    def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, max_iters: int = 200):
        fit, _ = nnls(basis, signal, maxiter=max_iters)
        return idx, fit

    def NNLSregCV(idx: int, signal: np.ndarray, basis: np.ndarray, tol: float = 0.0001):
        fit, _, _ = NNLSregCV(basis, signal, tol)
        return idx, fit

    # Not working with CurveFit atm
    # def model_multi_exp(nComponents: int):
    #     def model(bValues: np.ndarray, X: np.ndarray):
    #         function = 0
    #         for ii in range(
    #             nComponents - 2
    #         ):  # for 1 component the idx gets negative and for is evaded
    #             function = +np.array(
    #                 np.exp(-np.kron(bValues, abs(X[ii + 1]) * X[nComponents + ii + 1]))
    #             )
    #         return X[0] * (
    #             function
    #             + np.array(
    #                 np.exp(-np.kron(bValues, abs(X[nComponents])))
    #                 * (1 - np.sum(X[nComponents + 1 : -1]))
    #             )
    #         )

    #     return model
    def model_mono(bValues: np.ndarray, S0, D):
        return np.array(S0 * np.exp(-np.kron(bValues, D)))

    def monoFit(
        idx: int,
        signal: np.ndarray,
        bValues: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        fit, temp = curve_fit(
            FitModels.model_mono,
            bValues,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit

    def model_mono_t1(TM: int):
        def model(
            bValues: np.ndarray, S0: float | int, D: float | int, T1: float | int
        ):
            return np.array(S0 * np.exp(-np.kron(bValues, D)) * np.exp(-T1 / TM))

        return model

    def mono_t1Fit(
        idx: int,
        signal: np.ndarray,
        bValues: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        TM: int,
    ):
        fit, _ = curve_fit(
            FitModels.model_mono_t1(TM=TM),
            bValues,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit


def setup_pixelwise_fitting(fit_data, debug: bool | None = False) -> Nii:
    # prepare Workers
    img = fit_data.img
    mask = fit_data.mask
    fit_params = fit_data.fit_params

    if debug:
        pixel_args = zip(
            (
                ((i, j, k), img.array[i, j, k, :])
                for i, j, k in zip(*np.nonzero(np.squeeze(mask.array, axis=3)))
            ),
        )
    else:
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(mask.array, axis=3)))),
            (
                img.array[i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(mask.array, axis=3)))
            ),
        )
    if (
        fit_params.fit_model == FitModels.NNLS
        or fit_params.fit_model == FitModels.NNLSregCV
    ):
        # Prepare basis for NNLS from bValues and
        basis = np.exp(
            -np.kron(
                fit_params.bValues.T,
                fit_params.get_DValues(),
            )
        )
        fitfunc = partial(fit_params.fit_model, basis=basis)
    elif fit_params.fit_model == FitModels.monoFit:
        basis = fit_params.bValues
        fitfunc = partial(
            fit_params.fit_model,
            bValues=basis,
            x0=fit_params.Bounds.x0,
            lb=fit_params.Bounds.lb,
            ub=fit_params.Bounds.ub,
        )
    elif fit_params.fit_model == FitModels.mono_t1Fit:
        basis = fit_params.bValues
        fitfunc = partial(
            fit_params.fit_model,
            bValues=basis,
            x0=fit_params.Bounds.x0,
            lb=fit_params.Bounds.lb,
            ub=fit_params.Bounds.ub,
            TM=fit_params.variables.TM,
        )

    pixel_results = fit(fitfunc, pixel_args, fit_params.nPools, debug)

    # Sort Results
    fit_results = fit_data._fit_results()
    if (
        fit_params.fit_model == FitModels.NNLS
        or fit_params.fit_model == FitModels.NNLSregCV
    ):
        # Create output array for spectrum
        new_shape = np.array(mask.array.shape)
        new_shape[3] = basis.shape[1]
        fit_results = np.zeros(new_shape)
        # Sort Entries to array
        for pixel in pixel_results:
            fit_results[pixel[0]] = pixel[1]
        # TODO: add Ds and Fs
    elif fit_params.fit_model == FitModels.monoFit:
        fit_results.S0s = [None] * len(list(pixel_results))
        fit_results.Ds = [None] * len(list(pixel_results))
        fit_results.Fs = [None] * len(list(pixel_results))
        for idx, pixel in enumerate(pixel_results):
            fit_results.S0s[idx] = (pixel[0], np.array([pixel[1][0]]))
            fit_results.Ds[idx] = (pixel[0], np.array([pixel[1][1]]))
            fit_results.Fs[idx] = (pixel[0], np.array([1]))
        fit_data.fit_results = fit_results
        fit_data.set_spectrum_from_variables()
    elif fit_params.fit_model == FitModels.mono_t1Fit:
        fit_results.S0s = [None] * len(list(pixel_results))
        fit_results.Ds = [None] * len(list(pixel_results))
        fit_results.T1s = [None] * len(list(pixel_results))
        fit_results.Fs = [None] * len(list(pixel_results))
        for idx, pixel in enumerate(pixel_results):
            fit_results.S0s[idx] = (pixel[0], np.array([pixel[1][0]]))
            fit_results.Ds[idx] = (pixel[0], np.array([pixel[1][1]]))
            fit_results.T1s[idx] = (pixel[0], np.array([pixel[1][2]]))
            fit_results.Fs[idx] = (pixel[0], np.array([1]))
        fit_data.fit_results = fit_results
        fit_data.set_spectrum_from_variables()
    # Create output
    return Nii().from_array(fit_data.fit_results.spectrum)
    # return fit_results


def setup_signalbased_fitting(fit_data: fitData):
    img = fit_data.img
    seg = fit_data.mask
    fit_results = list()
    for seg_idx in range(1, seg.number_segs + 1, 1):
        img_seg = seg.get_single_seg_mask(seg_idx)
        signal = Processing.get_mean_seg_signal(img, img_seg, seg_idx)
        fit_results.append(fit_segmentation_signal(signal, fit_data, seg_idx))
    if fit_data.fit_params.fit_model == FitModels.NNLS:
        # Create output array for spectrum
        new_shape = np.array(seg.array.shape)
        basis = np.exp(
            -np.kron(
                fit_data.fit_params.bValues.T,
                fit_data.fit_params.get_DValues(),
            )
        )
        new_shape[3] = basis.shape[1]
        img_results = np.zeros(new_shape)
        # Sort Entries to array
        for seg in fit_results:
            img_results[seg[0]] = seg[1]


def fit_segmentation_signal(
    signal: np.ndarray, fit_params: fitData.fitParameters, seg_idx: int
):
    if fit_params.fit_model == FitModels.NNLS:
        basis = np.exp(
            -np.kron(
                fit_params.bValues.T,
                fit_params.get_DValues(),
            )
        )
        fit_function = partial(fit_params.fit_model, basis=basis)
    elif fit_params.fit_model == FitModels.monoFit:
        print("test")
    return fit_function(seg_idx, signal)


def fit(fitfunc, pixel_args, nPools, debug: bool | None = False):
    # Run Fitting
    if debug:
        pixel_results = []
        for pixel in pixel_args:
            pixel_results.append(fitfunc(pixel[0][0], pixel[0][1]))
    else:
        if nPools != 0:
            with Pool(nPools) as pool:
                pixel_results = pool.starmap(fitfunc, pixel_args)
    return pixel_results
