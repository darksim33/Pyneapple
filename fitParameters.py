import numpy as np
from scipy import signal
from scipy.sparse import diags
from functools import partial

from fitModel import Model
from utils import Nii, Nii_seg


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
    # NOTE: add find_peaks? Or where does Results get NNLS diff params from?


class Parameters:
    def __init__(
        self,
        model: Model,
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
        max_iter: int | None = None,
        n_pools: int | None = 4,  # cpu_count(),
    ):
        self.model = model
        self.b_values = b_values
        self.max_iter = max_iter
        self.boundaries = self._Boundaries()
        # self.TM = TM
        self.n_pools = n_pools
        self.fit_area = "Pixel"  # Pixel or Segmentation

    # NOTE: move/adjust _Boundaries == NNLSParams/MonoParams
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
        img: np.ndarray,
        seg: np.ndarray,
    ):
        # zip of tuples containing a tuple and a nd.array
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
            (
                img[i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
        )
        return pixel_args


class NNLSParams(Parameters):
    def __init__(
        self,
        model: Model | None = Model.NNLS,
        max_iter: int | None = 250,
        n_bins: int | None = 250,
        d_range: np.ndarray | None = np.array([1 * 1e-4, 2 * 1e-1]),
    ):
        """
        Basic NNLS Parameter Class
        model: should be of class Model
        """

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

    # TODO: ask ourselfs, if this is nested reasonable (move to results?)
    def eval_pixelwise_fitting_results(self, results, seg = Nii_seg) -> Results:
        # Create output array for spectrum
        spectrum_shape = np.array(seg.array.shape)
        spectrum_shape[3] = self.get_basis().shape[1]
        # Alternative line of code:
        #spectrum_shape = np.array(seg.array[..., None]) + self.get_basis()

        fit_results = Results()
        fit_results.spectrum = np.zeros(spectrum_shape)
        # Sort entries to array
        for pixel in results:
            fit_results.spectrum[pixel[0]] = pixel[1]
        return fit_results
    
    def eval_mean_fitting_results(self, results, seg) -> Results:
        # Create output array for spectrum
        new_shape = np.array(seg.array.shape)
        new_shape[3] = self.get_basis().shape[1]
        fit_results = Results()
        fit_results.spectrum = np.zeros(new_shape)
        # In case of multiple segmentations per slice
        for segmentation in results:
            fit_results.spectrum[segmentation[0]] = segmentation[1]
        return fit_results


class NNLSregParams(NNLSParams):
    def __init__(
        self,
        model,
        reg_order: int | None = 2,
        mu: float | None = 0.01,
    ):
        super().__init__(
            model=model,
            max_iter=100000,  # TODO ????? WHY
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
        img: np.ndarray,
        seg: np.ndarray,
    ):
        # enhance image array for regularisation
        reg = np.zeros((np.append(np.array(img.shape[0:3]), 250)))
        img_reg = np.concatenate((img, reg), axis=3)

        pixel_args = super().get_pixel_args(img_reg, seg)

        return pixel_args


class NNLSregCVParams(NNLSParams):
    def __init__(
        self, model: Model | None = Model.NNLS_reg_CV, tol: float | None = 0.0001
    ):
        super().__init__(model=model)
        self.tol = tol


class MonoParams(Parameters):
    def __init__(
        self,
        model: Model | None = Model.mono,
        x0: np.ndarray | None = np.array([50, 0.001]),
        lb: np.ndarray | None = np.array([10, 0.0001]),
        ub: np.ndarray | None = np.array([1000, 0.01]),
        TM: int | None = None,
        max_iter: int | None = 600,
    ):
        super().__init__(model=model, max_iter=max_iter)
        self.boundaries.x0 = x0
        self.boundaries.lb = lb
        self.boundaries.ub = ub
        self.TM = TM

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
            TM=self.TM,
            max_iter=self.max_iter,
        )

    def eval_pixelwise_fitting_results(self, results, seg) -> Results:
        # prepare arrays
        fit_results = Results()
        for pixel in results:
            fit_results.S0.append((pixel[0], [pixel[1][0]]))
            fit_results.d.append((pixel[0], [pixel[1][1]]))
            fit_results.f.append((pixel[0], np.ones(1)))

        fit_results = self.set_spectrum_from_variables(fit_results, seg)

        return fit_results

    def set_spectrum_from_variables(self, fit_results: Results, seg: Nii_seg):
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
        lb: np.ndarray | None = np.array([10, 0.0001, 1000]),
        ub: np.ndarray | None = np.array([1000, 0.01, 2500]),
        TM: float | None = 20.0,
        max_iter: int | None = 600,
    ):
        super().__init__(model=model, max_iter=max_iter)
        self.boundaries.x0 = x0
        self.boundaries.lb = lb
        self.boundaries.ub = ub
        self.TM = TM

    # NOTE: check inputs // matlab ideal
    # def get_fit_function(self):
    #     return partial(
    #         self.model,
    #         b_values=self.get_basis(),
    #         x0=self.boundaries.x0,
    #         lb=self.boundaries.lb,
    #         ub=self.boundaries.ub,
    #         TM=self.TM,
    #         max_iter=self.max_iter,
    #     )

    def eval_pixelwise_fitting_results(self, results, seg) -> Results:
        fit_results = super().eval_pixelwise_fitting_results(results, seg)
        # add aditional T1 results
        for pixel in results:
            fit_results.T1.append((pixel[0], [pixel[1][2]]))
        return fit_results
