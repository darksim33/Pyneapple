import numpy as np
from scipy import signal
from scipy.sparse import diags
from functools import partial
from typing import Callable

from .model import Model
from src.utils import NiiSeg


class Results:
    """
    Class containing estimated diffusion values and fractions

    ...

    Attributes
    ----------

    spectrum :

    d : dict()
    dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the d values
    f : list()
    dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the f values
    S0 : list()
    dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the S0 values
    T1 : list()
    dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the T1 values

    """

    def __init__(self):
        self.spectrum: np.ndarray = np.array([])
        self.raw: dict | np.ndarray = dict()
        self.d: dict | np.ndarray = dict()
        self.f: dict | np.ndarray = dict()
        self.S0: dict | np.ndarray = dict()
        self.T1: dict | np.ndarray = dict()
        # these should be lists of lists for each parameter

    # NOTE parameters lists of tuples containing
    # NOTE: add find_peaks? Or where does Results get NNLS diff params from?


class Parameters:
    def __init__(
        self,
        # model: Model.MultiExp | Model.NNLS | Model.NNLSregCV | Callable = None,
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
        self.b_values = b_values
        self.max_iter = max_iter
        self.boundaries = self._Boundaries()
        self.n_pools = n_pools
        self.fit_area = "Pixel"  # Pixel or Segmentation
        self.fit_model = None
        self.fit_function = None

    @property
    def fit_model(self):
        return self._fit_model

    @fit_model.setter
    def fit_model(self, value):
        self._fit_model = value

    @property
    def fit_function(self):
        return self._fit_function

    @fit_function.setter
    def fit_function(self, value):
        self._fit_function = value

    # NOTE: move/adjust _Boundaries == NNLSParams
    class _Boundaries:
        def __init__(
            self,
            lb: np.ndarray | None = np.array([]),  # lower bound
            ub: np.ndarray | None = np.array([]),  # upper bound
            x0: np.ndarray | None = np.array([]),  # starting values
            # TODO: relocate n_bins? not a boundary parameter
            n_bins: int | None = 250,
            d_range: np.ndarray
            | None = np.array(
                [1 * 1e-4, 2 * 1e-1]
            ),  # Lower and Upper Diffusion value for Range
        ):
            # needs fixing based on model maybe change according to model
            if lb.any():
                self.lb = lb
            if ub.any():
                self.ub = ub
            if x0.any():
                self.x0 = x0
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
            self.b_values = np.array([int(x) for x in f.read().split("\n")])

    def get_pixel_args(
        self,
        img: np.ndarray,
        seg: np.ndarray,
    ):
        # zip of tuples containing a tuple and a nd.array
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
            (img[i, j, k, :] for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
        )
        return pixel_args

    def eval_fitting_results(self, results, seg):
        pass


class NNLSParams(Parameters):
    def __init__(
        self,
        max_iter: int | None = 250,
        n_bins: int | None = 250,
        d_range: np.ndarray | None = np.array([1 * 1e-4, 2 * 1e-1]),
    ):
        """
        Basic NNLS Parameter Class
        model: should be of class Model
        """

        super().__init__(max_iter=max_iter)
        self.boundaries.n_bins = n_bins
        self.boundaries.d_range = d_range
        self._basis = np.array([])
        self.fit_function = Model.NNLS.fit

    @property
    def fit_function(self):
        return partial(self._fit_function, basis=self.get_basis())

    @fit_function.setter
    def fit_function(self, method: Callable):
        self._fit_function = method

    def get_basis(self) -> np.ndarray:
        self._basis = np.exp(
            -np.kron(
                self.b_values.T,
                self.get_bins(),
            )
        )
        return self._basis

    def eval_fitting_results(self, results, seg: NiiSeg) -> Results:
        # Create output array for spectrum
        spectrum_shape = np.array(seg.array.shape)
        spectrum_shape[3] = self.get_basis().shape[1]

        fit_results = Results()
        fit_results.spectrum = np.zeros(spectrum_shape)
        # Sort entries to array
        for element in results:
            fit_results.spectrum[element[0]] = element[1]
        return fit_results


class NNLSregParams(NNLSParams):
    def __init__(
        self,
        reg_order: int | None = 0,
        mu: float | None = 0.01,
    ):
        super().__init__(
            max_iter=100000,  # TODO ????? WHY
        )
        self.reg_order = reg_order
        self.mu = mu

    def get_basis(self) -> np.ndarray:
        basis = super().get_basis()
        n_bins = self.boundaries.n_bins

        if self.reg_order == 0:
            # no weighting
            reg = diags([1], [0], (n_bins, n_bins)).toarray()
        elif self.reg_order == 1:
            # weighting with the predecessor
            reg = diags([-1, 1], [0, 1], (n_bins, n_bins)).toarray()
        elif self.reg_order == 2:
            # weighting of the nearest neighbours
            reg = diags([1, -2, 1], [-1, 0, 1], (n_bins, n_bins)).toarray()
        elif self.reg_order == 3:
            # weighting of the first- and second-nearest neighbours
            reg = diags([1, 2, -6, 2, 1], [-2, -1, 0, 1, 2], (n_bins, n_bins)).toarray()
        else:
            raise NotImplemented(
                "Currently only supports regression orders of 3 or lower"
            )

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
        self,
        tol: float | None = 0.0001,
        reg_order: int | str | None = "CV",
    ):
        super().__init__()
        self.tol = tol
        self.reg_order: reg_order
        self.fit_function = Model.NNLSregCV.fit


class MultiExpParams(Parameters):
    """
    Properties:
    model(Callable): Property holding basic fitting model
    fit_model(Callable): Property holding actual fit method
    """
    def __init__(
        self,
        x0: np.ndarray | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
        TM: float | None = None,
        max_iter: int | None = 600,
        n_components: int | None = 3,
    ):
        super().__init__(max_iter=max_iter)
        self.max_iter = max_iter
        self.boundaries.x0 = x0
        self.boundaries.lb = lb
        self.boundaries.ub = ub
        self.TM = TM
        self.n_components = n_components
        self.fit_function = Model.MultiExp.fit
        self.fit_model = Model.MultiExp.wrapper
        if not x0:
            self.set_boundaries()

    @property
    def n_components(self):
        return self._n_components

    @property
    def fit_function(self):
        return partial(
            self._fit_function,
            b_values=self.get_basis(),
            args=self.boundaries.x0,
            lb=self.boundaries.lb,
            ub=self.boundaries.ub,
            n_components=self.n_components,
            TM=self.TM,
            max_iter=self.max_iter,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        self._fit_function = method

    @property
    def fit_model(self):
        return self._fit_model(n_components=self.n_components, TM=self.TM)

    @fit_model.setter
    def fit_model(self, method):
        self._fit_model = method

    @n_components.setter
    def n_components(self, value: int | str):
        if type(value) == str:
            if "MonoExp" in value:
                value = 1
            elif "BiExp" in value:
                value = 2
            elif "TriExp" in value:
                value = 3
        self._n_components = value
        if self.boundaries.x0 is None or not len(self.boundaries.x0) == value:
            self.set_boundaries()

    def set_boundaries(self):
        if self.n_components == 3:
            self.boundaries.x0 = np.array(
                [
                    0.0005,  # D_slow
                    0.01,  # D_inter
                    0.1,  # D_fast
                    0.3,  # f_slow
                    0.5,  # f_inter
                    210,  # S_0
                ]
            )
            self.boundaries.lb = np.array(
                [
                    0.0001,  # D_slow
                    0.003,  # D_inter
                    0.02,  # D_fast
                    0.01,  # f_slow
                    0.01,  # f_inter
                    10,  # S_0
                ]
            )
            self.boundaries.ub = np.array(
                [
                    0.003,  # D_slow
                    0.02,  # D_inter
                    0.4,  # D_fast
                    1,  # f_slow
                    1,  # f_inter
                    1000,  # S_0
                ]
            )
        elif self.n_components == 2:
            self.boundaries.x0 = np.array(
                [
                    0.0005,  # D_slow
                    0.01,  # D_inter
                    0.3,  # f_slow
                    210,  # S_0
                ]
            )
            self.boundaries.lb = np.array(
                [
                    0.0001,  # D_slow
                    0.003,  # D_inter
                    0.01,  # f_fast
                    10,  # S_0
                ]
            )
            self.boundaries.ub = np.array(
                [
                    0.003,  # D_slow
                    0.4,  # D_inter
                    1,  # f_fast
                    1000,  # S_0
                ]
            )
        elif self.n_components == 1:
            self.boundaries.x0 = np.array(
                [
                    0.005,  # D_slow
                    210,  # S_0
                ]
            )
            self.boundaries.lb = np.array(
                [
                    0.0001,  # D_slow
                    10,  # S_0
                ]
            )
            self.boundaries.ub = np.array(
                [
                    0.4,  # D_slow
                    1000,  # S_0
                ]
            )

    def get_basis(self):
        return np.squeeze(self.b_values)

    def eval_fitting_results(self, results, seg) -> Results:
        # prepare arrays
        fit_results = Results()
        for element in results:
            fit_results.raw[element[0]] = element[1]
            fit_results.S0[element[0]] = element[1][-1]
            fit_results.d[element[0]] = element[1][0 : self.n_components]
            f_new = np.zeros(self.n_components)
            f_new[: self.n_components - 1] = element[1][self.n_components : -1]
            f_new[-1] = 1 - np.sum(element[1][self.n_components : -1])
            fit_results.f[element[0]] = f_new

        # add additional T1 results if necessary
        if self.TM:
            for element in results:
                fit_results.T1[element[0]] = [element[1][2]]

        fit_results = self.set_spectrum_from_variables(fit_results, seg)

        return fit_results

    def set_spectrum_from_variables(self, fit_results: Results, seg: NiiSeg):
        # adjust d-values according to bins/d-values
        d_values = self.get_bins()

        # Prepare spectrum for dyn
        new_shape = np.array(seg.array.shape)
        new_shape[3] = self.boundaries.n_bins
        spectrum = np.zeros(new_shape)

        for pixel_pos in fit_results.d:
            temp_spec = np.zeros(self.boundaries.n_bins)
            d_new = list()
            for (D, F) in zip(fit_results.d[pixel_pos], fit_results.f[pixel_pos]):
                index = np.unravel_index(
                    np.argmin(abs(d_values - D), axis=None),
                    d_values.shape,
                )[0].astype(int)
                d_new.append(d_values[index])
                temp_spec = temp_spec + F * signal.unit_impulse(
                    self.boundaries.n_bins, index
                )
                spectrum[pixel_pos] = temp_spec
        fit_results.spectrum = spectrum
        return fit_results
