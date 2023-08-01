import numpy as np
from scipy.optimize import least_squares, curve_fit, nnls
from scipy import signal
from scipy.sparse import diags

# from typing import Callable
from multiprocessing import Pool, cpu_count
from functools import partial

from utils import Nii, Nii_seg, Processing
from fitting.NNLSregCV import NNLSregCV

from fit import fit
from model import Model


class FitData:
    def __init__(self, model, img: Nii | None = None, seg: Nii | None = None):
        self.model_name: str | None = model
        self.img = img if img is not None else Nii()  # same thing different syntax @TT?
        self.seg = seg if seg is not None else Nii_seg()
        self.fit_results = self.Results()
        if model == "NNLS":
            self.fit_params = NNLSParams(Model.NNLS)
        elif model == "NNLSreg":
            self.fit_params = NNLSregParams(Model.NNLS)
        elif model == "NNLSregCV":
            self.fit_params = NNLSregCVParams(Model.NNLS_reg_CV)
        elif model == "mono":
            self.fit_params = MonoParams(Model.mono)  # changed from MonoParams
        elif model == "mono_T1":
            self.fit_params = MonoT1Params(Model.mono)
        else:
            print("Error: no valid Algorithm")

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
            self.d: list | np.ndarray = list()
            self.f: list | np.ndarray = list()
            self.S0: list | np.ndarray = list()
            self.T1: list | np.ndarray = list()
            # these should be lists of lists for each parameter
            # NOTE paramters lists of tuples containing

    class Parameters:
        def __init__(
            self,
            model: Model
            | None = None,  # Wieso lasssen wir nochmal model = None zu? @TT
            b_values: np.ndarray
            | None = np.array(
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
            ),
            max_iter: int | None = 250,
            n_pools: int | None = 4,  # cpu_count(),
        ):
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

        class _Variables: # Why not in parameters @TT?
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

        def get_pixel_args(
                            self, 
                            img: Nii, 
                            seg: Nii_seg, 
                            # debug: bool
                            ):
            # if debug:
            #     pixel_args = zip(
            #         (
            #             ((i, j, k), img.array[i, j, k, :])
            #             for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
            #         )
            #     )
            # else:
            #     pixel_args = zip(
            #         (
            #             (i, j, k)
            #             for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
            #         ),
            #         (
            #             img.array[i, j, k, :]
            #             for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
            #         ),
            #     )
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
        fit_function = self.fit_params.get_fit_function()
        results_pixel = fit(fit_function, pixel_args, self.fit_params.n_pools, debug)
        self.fit_results = self.fit_params.eval_pixelwise_fitting_results(
            results_pixel, self.seg
        )


class NNLSParams(FitData.Parameters):
    def __init__(
        self,
        # TODO: model & b_values should be inherited without additional initialisation
        model: Model | None = Model.NNLS,
        max_iter: int
        | None = 250,  # TODO: necessary? only for re-initialising model...
        n_bins: int | None = 250,
        d_range: np.ndarray | None = np.array([1 * 1e-4, 2 * 1e-1]),
        # n_pools: int | None = 4,
    ):
        """
        Basic NNLS Parameter Class
        model: should be of class Model
        """
        # super()__init__(model, max_iter) # TODO: use this?
        if not model:
            super().__init__(Model.NNLS)
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

    def get_fit_function(self):
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
        model,
        reg_order: int
        | None = 2,  # TODO: fuse NNLSParams (reg=0) and NNLSregParams (reg!=0)?
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

    def get_pixel_args(
                        self, 
                        img: Nii, 
                        seg: Nii_seg, 
                        # debug: bool
                        ):
        # enhance image array for regularisation
        reg = np.zeros((np.append(np.array(img.array.shape[0:3]), 250)))
        img_reg = np.concatenate((img.array, reg), axis=3)

        # TODO: understand code @TT @JJ me neither
        # NOTE: Changed debug and normal to work the same way
        # if debug: # packing data for sequential for loop
        #     pixel_args = zip(
        #         (
        #             ((i, j, k), img_reg[i, j, k, :])
        #             for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
        #         )
        #     )
        # else:
        #     # Packing data for starmap multi threating
        #     # each element in the zip contains the adress (i, j, k) 
        #     pixel_args = zip(
        #         (
        #             (i, j, k)
        #             for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
        #         ),
        #         (
        #             img_reg[i, j, k, :]
        #             for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
        #         ),
        #     )

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
        self, model: Model | None = Model.NNLS_reg_CV, tol: float | None = 0.0001
    ):
        super().__init__(model=model)
        self.tol = tol


class MonoParams(FitData.Parameters):
    def __init__(
        self,
        model: Model | None = Model.mono, # unnötige Abfrage? @TT
        x0: np.ndarray | None = np.array([50, 0.001]),
        lb: np.ndarray | None = np.array([10, 0.0001]),
        ub: np.ndarray | None = np.array([1000, 0.01]),
        TM: int | None = None,
        max_iter: int
        | None = 600,  # TODO: None überflüssig? schon in Parameters gesetzt
    ):
        super().__init__(model=model, max_iter=max_iter)
        self.boundaries.x0 = x0
        self.boundaries.lb = lb
        self.boundaries.ub = ub
        self.variables.TM = TM

    # why function and not just self.b_values? @TT
    def get_basis(self):
        # BUG Bvlaues are passed in the wrong shape
        return np.squeeze(self.b_values)

    def get_fit_function(self):
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
        # prepare arrays
        fit_results = FitData.Results()
        for pixel in results_pixel:
            fit_results.S0.append((pixel[0], [pixel[1][0]]))
            fit_results.d.append((pixel[0], [pixel[1][1]]))
            fit_results.f.append((pixel[0], np.ones(1)))

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
        model: Model | None = Model.mono,
        x0: np.ndarray | None = np.array([50, 0.001, 1750]),
        lb: np.ndarray | None =  np.array([10, 0.0001, 1000]),
        ub: np.ndarray | None = np.array([1000, 0.01, 2500]),
        TM: float | None = 20.0,
        max_iter: int | None = 600,
    ):
        super().__init__(model=model, max_iter=max_iter)
        self.boundaries.x0 = x0
        self.boundaries.lb = lb 
        self.boundaries.ub = ub 
        self.variables.TM = TM 

    # TODO: same as in MonoT1 and NNLS -> inherit functions?
    def get_fit_function(self):
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
            fit_results.T1.append((pixel[0], [pixel[1][2]]))
        return fit_results