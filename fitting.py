import numpy as np
from scipy.optimize import least_squares, curve_fit, nnls
from scipy import signal
from typing import Callable
from multiprocessing import Pool, cpu_count
from functools import partial

from utils import *
from fromMedia.NNLSreg import NNLSreg


class fitData:
    def __init__(
        self, modelName, img: nifti_img | None = None, mask: nifti_img | None = None
    ):
        self.modelName: str | None = modelName
        self.img: nifti_img | None = img
        self.mask: nifti_img | None = mask
        self.fitParams = self.fitParameters()
        self.fitResults = self._fitResults()

    class _fitResults:
        """
        Class containing Diffusion values and Fractions
        """

        def __init__(self):
            self.spectrum: np.ndarray
            self.Ds: np.ndarray
            self.Fs: np.ndarray

    class fitParameters:
        def __init__(
            self,
            fitModel: Callable | None = None,
            bValues: np.ndarray | None = np.array([]),
            nPools: int = 4,  # cpu_count(),
        ):
            self.bValues = bValues
            self.fitModel = fitModel
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
                | None = None,  # Lower and Upper Diffusion value for Range
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

        def loadBvals(self, file: str):
            with open(file, "r") as f:
                # find away to decide which one is right
                # self.bvalues = np.array([int(x) for x in f.read().split(" ")])
                self.bValues = np.array([int(x) for x in f.read().split("\n")])

    def set_SpectrumFromVariables(self):
        # adjust D values acording to bins/dvalues
        DValues = self.fitParams.get_DValues()
        spectrum = np.zeros(self.fitParams.Bounds.nbins)
        DsNew = np.zeros(len(self.fitResults.Ds))
        for idx in range(len(self.fitResults.Ds)):
            index = np.unravel_index(
                np.argmin(abs(DValues - self.fitResults.Ds[idx]), axis=None),
                DValues.shape,
            )[0].astype(int)
            DsNew[idx] = DValues[index]
            spectrum = spectrum + self.fitResults.Fs[idx] * signal.unit_impulse(
                self.fitParams.Bounds.nbins, index
            )
        self.spectrum = spectrum


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
            super().__init__(fitModels.NNLS, bValues, nPools=nPools)
        elif model == "NNLSreg":
            super().__init__(fitModels.NNLSreg, bValues, nPools=nPools)
        self.Bounds.nbins = nbins
        self.Bounds.DiffBounds = DiffBounds


class MonoParams(fitData.fitParameters):
    def __init__(
        self,
        model: str | None = None,
        bValues: np.ndarray | None = np.array([]),
        nPools: int = 4,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
        x0: np.ndarray | None = None,
    ):
        if model == "mono":
            super().__init__(fitModels.model_multi_exp(1), bValues, nPools=nPools)
        elif model == "mono_t1":
            super().__init__(fitModels.mono_t1, bValues, nPools=nPools)
        self.Bounds.lb = lb
        self.Bounds.ub = ub
        self.Bounds.x0 = x0


class fitModels(object):
    def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, maxIters: int = 200):
        fit, _ = nnls(basis, signal, maxiter=maxIters)
        return idx, fit

    def NNLSreg(idx: int, signal: np.ndarray, basis: np.ndarray):
        fit, _, _ = NNLSreg(basis, signal)
        return idx

    def mono_t1(TM: int):
        def model(bvalues: np.ndarray, S0, D, T1):
            return np.array(S0 * np.exp(-np.kron(bvalues, D)) * np.exp(-T1 / TM))

        return model

    def model_multi_exp(nComponents: int):
        def model(bValues: np.ndarray, X: np.ndarray):
            function = np.array()
            for ii in range(
                nComponents - 2
            ):  # for 1 component the idx gets negative and for is evaded
                function = function + np.array(
                    np.exp(-np.kron(bValues, abs(X[ii + 1]) * X[nComponents + ii + 1]))
                )
            return X[0] * (
                function
                + np.array(
                    np.exp(-np.kron(bValues, abs(X[nComponents])))
                    * (1 - np.sum(X[nComponents + 1 : -1]))
                )
            )

        return model


def setupFitting(
    img: nifti_img,
    mask: nifti_img,
    fit_params: fitData.fitParameters,
    debug: bool = False,
) -> nifti_img:
    # prepare Workers
    # find data idx and prepare list of pixels to fit
    x = fit_params.get_DValues
    if debug:
        pixel_args = zip(
            (
                ((i, j, k), img.array[i, j, k, :])
                for i, j, k in zip(*np.nonzero(np.squeeze(mask.array)))
            ),
        )
    else:
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(mask.array)))),
            (
                img.array[i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(mask.array)))
            ),
        )
    if (
        fit_params.fitModel == fitModels.NNLS
        or fit_params.fitModel == fitModels.NNLSreg
    ):
        # Prepare basis for NNLS from bValues and
        basis = np.exp(
            -np.kron(
                fit_params.bValues.T,
                fit_params.get_DValues(),
            )
        )
        fit = partial(fit_params.fitModel, basis=basis)
    elif fit_params.fitModel == fitModels.model_multi_exp(1):
        basis = fit_params.bValues
        fit = partial(fit_params.fitModel, bvalues=basis)
    elif fit_params.fitModel == fitModels.monot1:
        basis = fit_params.bValues
        fit = partial(fit_params.fitModel(fit_params.variables.TM))

    # Run Fitting
    if debug:
        pixel_results = []
        for pixel in pixel_args:
            pixel_results.append(fit(pixel[0][0], pixel[0][1]))
    else:
        if fit_params.nPools != 0:
            with Pool(fit_params.nPools) as pool:
                pixel_results = pool.starmap(fit, pixel_args)

    # Create output array for spectrum
    new_shape = np.array(mask.array.shape)
    new_shape[3] = basis.shape[1]
    fit_results = np.zeros(new_shape)

    # Sort Entries to array
    for pixel in pixel_results:
        fit_results[pixel[0]] = pixel[1]

    # Create output
    return nifti_img().fromArray(fit_results)
