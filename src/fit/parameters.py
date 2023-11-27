import numpy as np
import math
from scipy import signal
from scipy.sparse import diags
from functools import partial
from typing import Callable
import json
from pathlib import Path
from abc import ABC, abstractmethod
import xlsxwriter

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
        self.spectrum: dict | np.ndarray = dict()
        self.curve: dict | np.ndarray = dict()
        self.raw: dict | np.ndarray = dict()
        self.d: dict | np.ndarray = dict()
        self.f: dict | np.ndarray = dict()
        self.S0: dict | np.ndarray = dict()
        self.T1: dict | np.ndarray = dict()

    def save_peaks_to_excel(self, file: Path):
        if self.raw:
            with xlsxwriter.Workbook(file) as workbook:
                worksheet = workbook.add_worksheet()
                worksheet.write_row(0, 0, ["index", "pixel_index", "d_value", "amp"])
                idx = 1
                for key in self.d:
                    for item_idx in range(len(self.d[key])):
                        worksheet.write_row(
                            idx,
                            0,
                            [
                                idx,
                                str(key),
                                self.d[key][item_idx],
                                self.f[key][item_idx],
                            ],
                        )
                        idx += 1
            print(f"Saved fit data to {file}")
        else:
            print("No fitted Data found.")


class Params(ABC):
    @property
    @abstractmethod
    def fit_function(self):
        pass

    @property
    @abstractmethod
    def fit_model(self):
        pass

    @abstractmethod
    def get_pixel_args(self, img, seg):
        pass

    @abstractmethod
    def eval_fitting_results(self, results, seg):
        pass


class Parameters(Params):
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
        self.boundaries = dict()
        self.boundaries["lb"] = np.array([])
        self.boundaries["ub"] = np.array([])
        self.boundaries["x0"] = np.array([])
        self.boundaries["n_bins"] = 250
        self.boundaries["d_range"] = np.array([1 * 1e-4, 2 * 1e-1])

        self.n_pools = n_pools
        self.fit_area = "Pixel"  # Pixel or Segmentation
        self.fit_model = lambda: None
        self.fit_function = lambda: None

    @property
    def b_values(self):
        return self._b_values

    @b_values.setter
    def b_values(self, values: np.ndarray | list):
        if type(values) == list:
            values = np.array(values)
        if type(values) == np.ndarray:
            self._b_values = np.expand_dims(values.squeeze(), axis=1)

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

    def get_bins(self) -> np.ndarray:
        """
        Returns range of Diffusion values for NNLS fitting or plotting
        """
        return np.array(
            np.logspace(
                np.log10(self.boundaries["d_range"][0]),
                np.log10(self.boundaries["d_range"][1]),
                self.boundaries["n_bins"],
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

    def apply_AUC_to_results(self, fit_results):
        return fit_results.d, fit_results.f

    @staticmethod
    def load_from_json(file_name: str | Path):
        with open(file_name, "r") as json_file:
            data_dict = json.load(json_file)
        if data_dict["Class"] in globals():
            fit_params = globals()[data_dict["Class"]]()
        else:
            print("Error: Wrong Class Header!")
            return
        for key, item in data_dict.items():
            entries = key.split(".")
            current_obj = fit_params
            if len(entries) > 1:
                for entry in entries[:-1]:
                    current_obj = getattr(current_obj, entry)
            if hasattr(current_obj, entries[-1]):
                # json Decoder
                if isinstance(item, list):
                    item = np.array(item)
                setattr(current_obj, entries[-1], item)
        return fit_params

    def save_to_json(self, file_path: Path):
        attributes = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and not attr.startswith("_")
            and not isinstance(getattr(self, attr), partial)
        ]
        data_dict = dict()
        data_dict["Class"] = self.__class__.__name__
        for attr in attributes:
            if attr == "boundaries":
                continue
            # Custom Encoder
            if isinstance(getattr(self, attr), np.ndarray):
                value = getattr(self, attr).squeeze().tolist()
            else:
                value = getattr(self, attr)
            data_dict[attr] = value
        if not file_path.exists():
            with file_path.open("w") as file:
                file.write("")
        with file_path.open("w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        print(f"Parameters saved to {file_path}")


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
        self.boundaries["n_bins"] = n_bins
        self.boundaries["d_range"] = d_range
        self._basis = np.array([])
        self.fit_function = Model.NNLS.fit
        self.fit_model = Model.NNLS.model

    @property
    def fit_function(self):
        return partial(self._fit_function, basis=self.get_basis())

    @fit_function.setter
    def fit_function(self, method: Callable):
        self._fit_function = method

    def get_basis(self) -> np.ndarray:
        self._basis = np.exp(
            -np.kron(
                self.b_values,
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

        # find peaks and calculate fractions
        bins = self.get_bins()
        for element in results:
            idx, properties = signal.find_peaks(element[1], height=0)
            f_values = properties["peak_heights"]
            # normalize f
            f_values = np.divide(f_values, sum(f_values))

            fit_results.d[element[0]] = bins[idx]
            fit_results.f[element[0]] = f_values

        # set curve
        for element in results:
            curve = self.fit_model(
                self.b_values,
                element[1],
                bins,
            )
            fit_results.curve[element[0]] = curve

        return fit_results

    def apply_AUC_to_results(self, fit_results) -> (dict, dict):
        regime_boundaries = [0.003, 0.05, 0.3]  # use d_range instead?
        n_regimes = len(regime_boundaries)
        d_AUC, f_AUC = {}, {}

        # Analyse all elements for application of AUC
        for (key, d_values), (_, f_values) in zip(
            fit_results.d.items(), fit_results.f.items()
        ):
            d_AUC[key] = np.zeros(n_regimes)
            f_AUC[key] = np.zeros(n_regimes)

            for idx, regime_boundary in enumerate(regime_boundaries):
                # Check for peaks inside regime
                peaks_in_regime = d_values < regime_boundary

                if not any(peaks_in_regime):
                    continue

                # Merge all peaks within this regime with weighting
                d_regime = d_values[peaks_in_regime]
                f_regime = f_values[peaks_in_regime]
                f_AUC[key][idx] = sum(f_regime)
                d_AUC[key][idx] = np.dot(d_regime, f_regime) / sum(f_regime)

                # Build set difference for analysis of left peaks
                d_values = np.setdiff1d(d_values, d_regime)
                f_values = np.setdiff1d(f_values, f_regime)

        return d_AUC, f_AUC


class NNLSregParams(NNLSParams):
    def __init__(
        self,
        reg_order: int | None = 0,
        mu: float | None = 0.02,
    ):
        super().__init__(max_iter=200)
        self.reg_order = reg_order
        self.mu = mu

    def get_basis(self) -> np.ndarray:
        basis = super().get_basis()
        n_bins = self.boundaries["n_bins"]

        if self.reg_order == 0:
            # no weighting
            # reg = diags([1], [0], (n_bins, n_bins)).toarray()
            reg = np.eye(n_bins)
        elif self.reg_order == 1:
            # weighting with the predecessor
            reg = diags([-1, 1], [0, 1], (n_bins, n_bins)).toarray() * self.mu
        elif self.reg_order == 2:
            # weighting of the nearest neighbours
            reg = diags([1, -2, 1], [-1, 0, 1], (n_bins, n_bins)).toarray() * self.mu
        elif self.reg_order == 3:
            # weighting of the first- and second-nearest neighbours
            reg = (
                diags([1, 2, -6, 2, 1], [-2, -1, 0, 1, 2], (n_bins, n_bins)).toarray()
                * self.mu
            )
        else:
            raise NotImplemented(
                "Currently only supports regression orders of 3 or lower"
            )

        # append reg to create regularised NNLS basis
        return np.concatenate((basis, reg))

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

    def eval_fitting_results(self, results, seg: NiiSeg) -> Results:
        # Create output array for spectrum
        spectrum_shape = np.array(seg.array.shape)
        spectrum_shape[3] = self.get_basis().shape[1]

        fit_results = Results()
        fit_results.spectrum = np.zeros(spectrum_shape)
        # Sort entries to array
        for element in results:
            fit_results.spectrum[element[0]] = element[1]

        # find peaks and calculate fractions
        bins = self.get_bins()
        for element in results:
            # find all peaks and corresponding d_values
            idx, properties = signal.find_peaks(element[1], height=0)
            d_values = bins[idx]

            # from the found peaks get heights and widths
            f_peaks = properties["peak_heights"]
            f_fwhms = signal.peak_widths(element[1], idx, rel_height=0.5)[0]
            f_values = list()
            for peak, fwhm in zip(f_peaks, f_fwhms):
                # calculate area under the curve fractions by assuming gaussian curve
                f_values.append(
                    np.multiply(peak, fwhm)
                    / (2 * math.sqrt(2 * math.log(2)))
                    * math.sqrt(2 * math.pi)
                )
            f_values = np.divide(f_values, sum(f_values))
            fit_results.d[element[0]] = d_values
            fit_results.f[element[0]] = f_values

        # set curve
        for element in results:
            curve = self.fit_model(
                self.b_values,
                element[1],
                bins,
            )
            fit_results.curve[element[0]] = curve

        return fit_results


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
        self.boundaries["x0"] = x0
        self.boundaries["lb"] = lb
        self.boundaries["ub"] = ub
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
            args=self.boundaries["x0"],
            lb=self.boundaries["lb"],
            ub=self.boundaries["ub"],
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
        if self.boundaries["x0"] is None or not len(self.boundaries["x0"]) == value:
            self.set_boundaries()

    def set_boundaries(self):
        if self.n_components == 3:
            self.boundaries["x0"] = np.array(
                [
                    0.0005,  # D_slow
                    0.01,  # D_inter
                    0.1,  # D_fast
                    0.3,  # f_slow
                    0.5,  # f_inter
                    210,  # S_0
                ]
            )
            self.boundaries["lb"] = np.array(
                [
                    0.0001,  # D_slow
                    0.003,  # D_inter
                    0.02,  # D_fast
                    0.01,  # f_slow
                    0.01,  # f_inter
                    10,  # S_0
                ]
            )
            self.boundaries["ub"] = np.array(
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
            self.boundaries["x0"] = np.array(
                [
                    0.0005,  # D_slow
                    0.01,  # D_inter
                    0.3,  # f_slow
                    210,  # S_0
                ]
            )
            self.boundaries["lb"] = np.array(
                [
                    0.0001,  # D_slow
                    0.003,  # D_inter
                    0.01,  # f_fast
                    10,  # S_0
                ]
            )
            self.boundaries["ub"] = np.array(
                [
                    0.003,  # D_slow
                    0.4,  # D_inter
                    1,  # f_fast
                    1000,  # S_0
                ]
            )
        elif self.n_components == 1:
            self.boundaries["x0"] = np.array(
                [
                    0.005,  # D_slow
                    210,  # S_0
                ]
            )
            self.boundaries["lb"] = np.array(
                [
                    0.0001,  # D_slow
                    10,  # S_0
                ]
            )
            self.boundaries["ub"] = np.array(
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
            # add curve fit
            fit_results.curve[element[0]] = self.fit_model(self.b_values, *element[1])

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
        new_shape[3] = self.boundaries["n_bins"]
        spectrum = np.zeros(new_shape)

        for pixel_pos in fit_results.d:
            temp_spec = np.zeros(self.boundaries["n_bins"])
            d_new = list()
            for D, F in zip(fit_results.d[pixel_pos], fit_results.f[pixel_pos]):
                index = np.unravel_index(
                    np.argmin(abs(d_values - D), axis=None),
                    d_values.shape,
                )[0].astype(int)
                d_new.append(d_values[index])
                temp_spec = temp_spec + F * signal.unit_impulse(
                    self.boundaries["n_bins"], index
                )
                spectrum[pixel_pos] = temp_spec
        fit_results.spectrum = spectrum
        return fit_results
