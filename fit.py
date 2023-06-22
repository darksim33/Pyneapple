import numpy as np
from scipy.optimize import least_squares, curve_fit, nnls
from scipy import signal

# from typing import Callable
from multiprocessing import Pool, cpu_count
from functools import partial

from utils import Nii, Nii_seg, Processing
from fitting.NNLSregCV import NNLSregCV


class FitData:
    def __init__(self, model, img: Nii | None = None, seg: Nii | None = None):
        self.model_name: str | None = model
        self.img = img if img is not None else Nii()
        self.seg = seg if seg is not None else Nii_seg()
        # self.fit_params = self.Parameters(model=None) # why commented? Rename "Parameters"?
        self.fit_results = self.Results()
        if model == "NNLS":
            self.fit_params = NNLSParams(Model.NNLS)
        elif model == "NNLSreg":
            self.fit_params = NNLSregParams(Model.NNLS_reg)
        elif model == "NNLSregCV":
            self.fit_params = NNLSParams(Model.NNLS_reg_CV)
        elif model == "mono":
            self.fit_params = MonoParams("mono")
        elif model == "mono_T1":
            self.fit_params == MonoParams("mono_T1")
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
            model: str | None = None,
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

            print(b_values.shape)
            self.model = model
            self.b_values = b_values
            self.boundaries = self._Boundaries()
            self.variables = self._Variables()
            self._nPools = nPools

        @property  # is this necessary @JJ?
        def nPools(self):
            return self._nPools

        @nPools.setter
        def nPools(self, number):
            self._nPools = number

        # _Boundaries == NNLSParams/MonoParams? @TT
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
                # # "neets"? U serious, @TT?
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

    def fitting_pixelwise(self, debug: bool = False):
        # TODO: add seg number utility
        pixel_args = self.fit_params.get_pixel_args(self.img, self.seg, debug)
        fit_function = self.fit_params.get_partial_fit_function()
        results_pixel = fit(fit_function, pixel_args, self.fit_params.nPools, debug)
        self.fit_results = self.fit_params.eval_pixelwise_fitting_results(
            results_pixel, self.seg
        )


# maybe use simplest model (mono) as standard for inheritance chain (based on Parameter class)? @TT
class NNLSParams(FitData.Parameters):
    def __init__(
        self,
        # TODO: inheritance fix, model & b_values should be inherited without additional initialisation
        model: None = None,
        # b_values: np.ndarray | None = None,
        n_bins: int | None = 250,
        d_range: np.ndarray | None = np.array([1 * 1e-4, 2 * 1e-1]),
        # nPools: int | None = 4,
    ):
        """Basic NNLS Parameter Class"""
        # why not: "if not b_values np.array(...)" ?

        if not model:
            super().__init__(Model.NNLSs)
        else:
            super().__init__(model)
        self.boundaries.n_bins = n_bins
        self.boundaries.d_range = d_range
        self._max_iter = 250

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter

    def get_basis(self) -> np.ndarray:
        self._basis = np.exp(
            -np.kron(
                self.b_values.T,
                self.get_d_values(),
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
        # model: str | None,
        # b_values: np.ndarray | None,
        # n_bins: int | None,
        # d_range: np.ndarray | None,
        # nPools: int | None = None,
        reg_order: int | None = 2,
        mu: float | None = 0.01,
    ):
        super().__init__(
            model=Model.NNLS_reg,
            # b_values=b_values,
            # n_bins=n_bins,
            # d_range=d_range,
            # nPools=nPools,
        )
        self._reg_order = reg_order
        self._mu = mu

    @property
    def reg_order(self):
        return self._reg_order

    @reg_order.setter
    def reg_order(self, reg_order):
        self._reg_order = reg_order

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    # TODO: fix inheritance, get_basis method already in NNLSParams
    # @JoJas102 basis differce between reg and basic version, therefore new function is needed
    def get_basis(self) -> np.ndarray:
        basis = np.exp(
            -np.kron(
                self.b_values.T,
                self.get_d_values(),
            )
        )
        n_data = self.b_values.shape[1]
        n_bins = self.boundaries.n_bins

        # create new basis and signal for reg fitting
        basis_reg = np.zeros([n_data + n_bins, n_bins])
        # add current set to new one
        basis_reg[0:n_data, 0:n_bins] = basis

        # TODO: simplify reg code?
        for i in range(n_bins, (n_bins + n_data), 1):
            # idx_data is iterator for the datapoints
            # since the reg basis is already filled with the basis set it only needs to iterate beyond that
            for j in range(n_bins):
                # idx_bin is the iterator for the bins
                basis_reg[i, j] = 0
                if self._reg_order == 0:
                    # no weighting
                    if i - n_data == j:
                        basis_reg[i, j] = 1.0 * self._mu
                elif self._reg_order == 1:
                    # weighting with the predecessor
                    if i - n_data == j:
                        basis_reg[i, j] = -1.0 * self._mu
                    elif i - n_data == j + 1:
                        basis_reg[i, j] = 1.0 * self._mu
                elif self._reg_order == 2:
                    # weighting of the nearest neighbours
                    if i - n_data == j - 1:
                        basis_reg[i, j] = 1.0 * self._mu
                    elif i - n_data == j:
                        basis_reg[i, j] = -2.0 * self._mu
                    elif i - n_data == j + 1:
                        basis_reg[i, j] = 1.0 * self._mu
                elif self._reg_order == 3:
                    # weighting of the first and second nearest neighbours
                    if i - n_data == j - 2:
                        basis_reg[i, j] = 1.0 * self._mu
                    elif i - n_data == j - 1:
                        basis_reg[i, j] = 2.0 * self._mu
                    elif i - n_data == j:
                        basis_reg[i, j] = -6.0 * self._mu
                    elif i - n_data == j + 1:
                        basis_reg[i, j] = 2.0 * self._mu
                    elif i - n_data == j + 2:
                        basis_reg[i, j] = 1.0 * self._mu

        return basis_reg

    def get_pixel_args(self, img: Nii, seg: Nii_seg, debug: bool):
        # enhance image array for regularisation
        img_new = np.zeros(
            [
                img.array.shape[0],
                img.array.shape[1],
                img.array.shape[2],
                (img.array.shape[3] + self.boundaries.n_bins),
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
                    img_new[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
                ),
            )
        return pixel_args

class NNLSregCVParams(NNLSParams):
    def __init__(self, tol: float | None = 0.0001):
        super().__init__(model = Model.NNLS_reg_CV)
        self._tol = tol
        
    @property
    def tol(self):
        return self._tol
    
    @tol.setter
    def tol(self, value):
        self._tol = value

class MonoParams(FitData.Parameters):
    def __init__(
        self,
        model: str | None = None,
        b_values: np.ndarray | None = np.array([]),
        nPools: int = 4,
        x0: np.ndarray | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
    ):
        if model == "mono":
            super().__init__(model=Model.mono_fit, b_values=b_values, nPools=nPools)
            self.boundaries.x0 = x0 if x0 is not None else np.array([50, 0.001])
            self.boundaries.lb = lb if lb is not None else np.array([10, 0.0001])
            self.boundaries.ub = ub if ub is not None else np.array([1000, 0.01])
        elif model == "mono_T1":
            super().__init__(model=Model.mono_T1_fit, b_values=b_values, nPools=nPools)
            self.boundaries.x0 = x0 if x0 is not None else np.array([50, 0.001, 1750])
            self.boundaries.lb = lb if lb is not None else np.array([10, 0.0001, 1000])
            self.boundaries.ub = ub if ub is not None else np.array([1000, 0.01, 2500])
        else:
            print("ERROR")

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


class Model(object):
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

    # Not working with CurveFit atm (for NLLS)
    # def model_multi_exp(nComponents: int):
    #     def model(b_values: np.ndarray, X: np.ndarray):
    #         function = 0
    #         for ii in range(
    #             nComponents - 2
    #         ):  # for 1 component the idx gets negative and for is evaded
    #             function = +np.array(
    #                 np.exp(-np.kron(b_values, abs(X[ii + 1]) * X[nComponents + ii + 1]))
    #             )
    #         return X[0] * (
    #             function
    #             + np.array(
    #                 np.exp(-np.kron(b_values, abs(X[nComponents])))
    #                 * (1 - np.sum(X[nComponents + 1 : -1]))
    #             )
    #         )

    #     return model

    # TODO: merge mono and mono_fit (same for *_T1)
    def mono(b_values: np.ndarray, S0, D):
        return np.array(S0 * np.exp(-np.kron(b_values, D)))

    def mono_fit(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        fit, temp = curve_fit(
            Model.mono,
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit

    def mono_T1(TM: int):
        def model(
            b_values: np.ndarray, S0: float | int, D: float | int, T1: float | int
        ):
            return np.array(S0 * np.exp(-np.kron(b_values, D)) * np.exp(-T1 / TM))

        return model

    def mono_T1_fit(
        idx: int,
        signal: np.ndarray,
        b_values: np.ndarray,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        TM: int,
    ):
        fit, _ = curve_fit(
            Model.mono_T1(TM=TM),
            b_values,
            signal,
            x0,
            bounds=(lb, ub),
        )
        return idx, fit


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
    if fit_params.model == Model.NNLS or fit_params.model == Model.NNLS_reg_CV:
        # Prepare basis for NNLS from b_values and
        basis = np.exp(
            -np.kron(
                fit_params.b_values.T,
                fit_params.get_d_values(),
            )
        )
        fitfunc = partial(fit_params.model, basis=basis)
    elif fit_params.model == Model.mono_fit:
        basis = fit_params.b_values
        fitfunc = partial(
            fit_params.model,
            b_values=basis,
            x0=fit_params.Bounds.x0,
            lb=fit_params.Bounds.lb,
            ub=fit_params.Bounds.ub,
        )
    elif fit_params.model == Model.mono_T1_fit:
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
    if fit_params.model == Model.NNLS or fit_params.model == Model.NNLS_reg_CV:
        # Create output array for spectrum
        new_shape = np.array(seg.array.shape)
        new_shape[3] = basis.shape[1]
        fit_results.spectrum = np.zeros(new_shape)
        # Sort Entries to array
        for pixel in results_pixel:
            fit_results.spectrum[pixel[0]] = pixel[1]
        # TODO: add d and f
    elif fit_params.model == Model.mono_fit:
        fit_results.S0 = [None] * len(list(results_pixel))
        fit_results.d = [None] * len(list(results_pixel))
        fit_results.f = [None] * len(list(results_pixel))
        for idx, pixel in enumerate(results_pixel):
            fit_results.S0[idx] = (pixel[0], np.array([pixel[1][0]]))
            fit_results.d[idx] = (pixel[0], np.array([pixel[1][1]]))
            fit_results.f[idx] = (pixel[0], np.array([1]))
        fit_data.fit_results = fit_results
        fit_data.set_spectrum_from_variables()
    elif fit_params.model == Model.mono_T1_fit:
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
    # Create output (why based on fit_data? @TT)
    return Nii().from_array(fit_data.fit_results.spectrum)


def setup_signalbased_fitting(fit_data: FitData):
    img = fit_data.img
    seg = fit_data.seg
    fit_results = list()
    for seg_idx in range(1, seg.number_segs + 1, 1):
        img_seg = seg.get_single_seg_mask(seg_idx)
        signal = Processing.get_mean_seg_signal(img, img_seg, seg_idx)
        fit_results.append(fit_segmentation_signal(signal, fit_data, seg_idx))
    if fit_data.fit_params.model == Model.NNLS:
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
    if fit_params.model == Model.NNLS:
        basis = np.exp(
            -np.kron(
                fit_params.b_values.T,
                fit_params.get_d_values(),
            )
        )
        fit_function = partial(fit_params.model, basis=basis)
    elif fit_params.model == Model.mono_fit:
        print("test")
    return fit_function(seg_idx, signal)


# does this function really fit?! Isn't fitting done in Models? Consider renaming @TT
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
