import numpy as np
import math
import json
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from scipy import signal
from scipy.sparse import diags
from functools import partial
from typing import Callable
from pathlib import Path
from abc import ABC, abstractmethod

from .model import Model
from ..utils.nifti import Nii, NiiSeg, NiiFit
from ..utils.exceptions import ClassMismatch
from ..utils.multithreading import multithreader, sort_interpolated_array


class Results:
    """
    Class containing estimated diffusion values and fractions.

    Attributes
    ----------
    d : dict
        Dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the d values
    f : list
        Dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the f values
    S0 : list
        Dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the S0 values
    T1 : list
        Dict of tuples containing pixel coordinates as keys and a np.ndarray holding all the T1 values

    Methods
    -------
    save_results(file_path, model)
        Creates results dict containing pixels position, slice number, fitted D and f values and total number of found
        compartments and saves it as Excel sheet.

    save_spectrum(file_path)
        Saves spectrum of fit for every pixel as 4D Nii.

    _set_up_results_struct(self, d=None, f=None):
        Sets up dict containing pixel position, slice, d, f and number of found compartments. Used in save_results
        function.

    create_heatmap(img_dim, model, d: dict, f: dict, file_path, slice_number=0)
        Creates heatmaps for d and f in the slices segmentation and saves them as PNG files. If no slice_number is
        passed, plots the first slice.
    """

    def __init__(self):
        self.spectrum: dict | np.ndarray = dict()
        self.curve: dict | np.ndarray = dict()
        self.raw: dict | np.ndarray = dict()
        self.d: dict | np.ndarray = dict()
        self.f: dict | np.ndarray = dict()
        self.S0: dict | np.ndarray = dict()
        self.T1: dict | np.ndarray = dict()

    def save_results_to_excel(self, file_path, d: dict = None, f: dict = None):
        """
        Saves the results of a model fit to an Excel file.

        Parameters
        ----------
        file_path : str
            The path where the Excel file will be saved.
        d : dict | None
            Optional argument. Sets diffusion coefficients to save if different from fit results.
        f : dict | None
            Optional argument. Sets volume fractions to save if different from fit results.
        """

        # Set d and f as current fit results if not passed
        if not (d or f):
            d = self.d
            f = self.f

        result_df = pd.DataFrame(self._set_up_results_struct(d, f)).T

        # Restructure key index into columns and save results
        result_df.reset_index(
            names=["pixel_x", "pixel_y", "slice", "compartment"], inplace=True
        )
        result_df.to_excel(file_path)

    def save_fitted_parameters_to_nii(
        self,
        file_path: str | Path,
        shape: tuple,
        parameter_names: list | dict | None = None,
        dtype: object | None = int,
    ):
        """
        Saves the results of a IVIM fit to an NifTi file.

        Parameters
        ----------
        file_path : str
            The path where the NifTi file will be saved.
        shape: np.ndarray
            Contains 3D matrix size of original Image
        dtype: type | None
            Handles datatype to save the NifTi in. int and float are supported.
        parameter_names: dict | None
            Containing the variables as keys and names as items (list of str)
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # determine number of parameters
        n_components = len(self.d[next(iter(self.d))])

        if len(shape) >= 4:
            shape = shape[:3]

        array = np.zeros((shape[0], shape[1], shape[2], n_components * 2 + 1))
        # Create Parameter Maps Matrix
        for key in self.d:
            array[key[0], key[1], key[2], 0:n_components] = self.d[key]
            array[key[0], key[1], key[2], n_components:-1] = self.f[key]
            array[key[0], key[1], key[2], -1] = self.S0[key]

        out_nii = NiiFit(n_components=n_components).from_array(array)

        # Check for Parameter Names
        if parameter_names is not None:
            if isinstance(parameter_names, dict):
                names = list()
                for key in parameter_names:
                    names = names + [key + item for item in parameter_names[key]]
            elif isinstance(parameter_names, list):
                names = parameter_names
            else:
                ValueError("Parameter names must be a dict or a list")
        else:
            names = None

        out_nii.save(
            file_path, dtype=dtype, save_type="separate", parameter_names=names
        )
        print("All files have been saved.")

    def save_spectrum_to_nii(self, file_path):
        """Saves spectrum of fit for every pixel as 4D Nii."""
        spec = Nii().from_array(self.spectrum)
        spec.save(file_path)

    def _set_up_results_struct(self, d=None, f=None):
        """
        Sets up dict containing pixel position, slice, d, f and number of found compartments.

        Parameters
        ----------
        d : dict | None
            Optional argument. Sets diffusion coefficients to save if different from fit results.
        f : dict | None
            Optional argument. Sets volume fractions to save if different from fit results.
        """

        # Set d and f as current fit results if not passed
        if not (d or f):
            d = self.d
            f = self.f

        result_dict = {}
        current_pixel = 0
        for key, d_values in d.items():
            n_comps = len(d_values)
            current_pixel += 1

            for comp, d_comp in enumerate(d_values):
                result_dict[key + (comp + 1,)] = {
                    "element": current_pixel,
                    "D": d_comp,
                    "f": f[key][comp],
                    "n_compartments": n_comps,
                }
        return result_dict

    @staticmethod
    def create_heatmap(fit_data, file_path, slices_contain_seg):
        """
        Creates heatmap plots for d and f results of pixels inside the segmentation, saved as PNG.

        N heatmaps are created dependent on the number of compartments up to the tri-exponential model.
        Needs d and f to be of same length throughout whole struct. Used in particular for AUC results.

        Parameters
        ----------
        fit_data : FitData
            FitData object holding model, img and seg information.
        file_path : str
            The path where the Excel file will be saved.
        slices_contain_seg : iterable
            Number of slice heatmap should be created of.
        """
        # Apply AUC (for smoothed results with >3 components)
        (d, f) = fit_data.fit_params.apply_AUC_to_results(fit_data.fit_results)
        img_dim = fit_data.img.array.shape[0:3]

        # Check first pixels result for underlying number of compartments
        n_comps = len(d[list(d)[0]])

        model = fit_data.model_name

        for slice_number, slice_contains_seg in enumerate(slices_contain_seg):
            if slice_contains_seg:
                # Create 4D array heatmaps containing d and f values
                d_heatmap = np.zeros(np.append(img_dim, n_comps))
                f_heatmap = np.zeros(np.append(img_dim, n_comps))

                for key, d_value in d.items():
                    d_heatmap[key + (slice(None),)] = d_value
                    f_heatmap[key + (slice(None),)] = f[key]

                # Plot heatmaps
                fig, axs = plt.subplots(2, n_comps)
                fig.suptitle(f"{model}", fontsize=20)

                for (param, comp), ax in np.ndenumerate(axs):
                    diff_param = [
                        d_heatmap[:, :, slice_number, comp],
                        f_heatmap[:, :, slice_number, comp],
                    ]

                    im = ax.imshow(np.rot90(diff_param[param]))
                    fig.colorbar(im, ax=ax, shrink=0.7)
                    ax.set_axis_off()

                fig.savefig(
                    Path(str(file_path) + "_" + model + f"_slice_{slice_number}.png")
                )


class BoundariesBase(ABC):
    @abstractmethod
    def load(self, _dict: dict):
        pass

    @abstractmethod
    def save(self) -> dict:
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> list | None:
        pass

    @property
    @abstractmethod
    def scaling(self):
        pass

    @abstractmethod
    def apply_scaling(self, value):
        pass

    @abstractmethod
    def get_axis_limits(self) -> tuple:
        pass


class Boundaries(BoundariesBase):
    def __init__(self):
        self.values = dict()
        self.scaling: str | int | float | list | None = None
        # a factor or string (needs to be added to apply_scaling to boundaries)
        self.dict = dict()
        self.number_points = 250

    def load(self, _dict: dict):
        self.dict = _dict.copy()

    def save(self):
        _dict = self.dict.copy()
        for key, value in _dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        value[sub_key] = sub_value.tolist()
            if isinstance(value, np.ndarray):
                _dict[key] = value.tolist()
        return _dict

    @property
    def parameter_names(self) -> list | None:
        """Returns parameter names from json for IVIM (and generic names vor NNLS)"""
        return None

    @parameter_names.setter
    def parameter_names(self, data: dict):
        pass

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    def apply_scaling(self, value: list) -> list:
        return value

    def get_axis_limits(self) -> tuple:
        return 0, 1


class Params(ABC):
    """Abstract base class for Parameters child class."""

    @property
    @abstractmethod
    def fit_function(self):
        pass

    @property
    @abstractmethod
    def fit_model(self):
        pass

    @property
    @abstractmethod
    def scale_image(self):
        pass

    @abstractmethod
    def get_pixel_args(self, img, seg, *args):
        pass

    @abstractmethod
    def get_seg_args(self, img: Nii, seg: NiiSeg, seg_number, *args):
        pass

    @abstractmethod
    def eval_fitting_results(self, results, seg):
        pass


class Parameters(Params):
    def __init__(self, params_json: str | Path | None = None):
        """
        Containing all relevant, partially model-specific parameters for fitting.

        Attributes
        ----------
        params_json: str | Path
            Json containing fitting parameters

        Methods
        -------
        ...
        """
        self.json = params_json
        # Set Basic Parameters
        self.b_values = None
        self.max_iter = None
        if not hasattr(self, "boundaries") or self.boundaries is None:
            self.boundaries = Boundaries()
        self.n_pools = None
        self.fit_area = None
        self.fit_model = lambda: None
        self.fit_function = lambda: None
        self.scale_image: str | None = None

        if isinstance(self.json, (str, Path)):
            self.json = Path(self.json)
            if self.json.is_file():
                self.load_from_json()
            else:
                print("Warning: Can't find parameter file!")
                self.json = None

    @property
    def b_values(self):
        return self._b_values

    @b_values.setter
    def b_values(self, values: np.ndarray | list | None):
        if isinstance(values, list):
            values = np.array(values)
        if isinstance(values, np.ndarray):
            self._b_values = np.expand_dims(values.squeeze(), axis=1)
        if values is None:
            self._b_values = None

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

    @property
    def scale_image(self):
        return self._scale_image

    @scale_image.setter
    def scale_image(self, value: str | int):
        self._scale_image = value
        self.boundaries.scaling = value

    def get_bins(self) -> np.ndarray:
        """Returns range of Diffusion values for NNLS fitting or plotting."""

        return np.array(
            np.logspace(
                np.log10(self.boundaries.get_axis_limits()[0]),
                np.log10(self.boundaries.get_axis_limits()[1]),
                self.boundaries.number_points,
            )
        )

    def load_b_values(self, file: str):
        """Loads b-values from json file."""
        with open(file, "r") as f:
            self.b_values = np.array([int(x) for x in f.read().split("\n")])

    def get_pixel_args(self, img: Nii, seg: NiiSeg, *args) -> zip:
        """Returns zip of tuples containing pixel arguments"""
        # zip of tuples containing a tuple and a nd.array
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))),
            (
                img.array[i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
            ),
        )
        return pixel_args

    def get_seg_args(self, img: Nii, seg: NiiSeg, seg_number: int, *args) -> zip:
        """Returns zip of tuples containing segment arguments"""
        mean_signal = seg.get_mean_signal(img.array, seg_number)
        return zip([[seg_number]], [mean_signal])

    def eval_fitting_results(self, results, seg):
        pass

    def apply_AUC_to_results(self, fit_results):
        return fit_results.d, fit_results.f

    def load_from_json(self, params_json: str | Path | None = None):
        if params_json:
            self.json = params_json

        with open(self.json, "r") as json_file:
            params_dict = json.load(json_file)

        # Check if .json contains Class identifier and if .json and Params set match
        if "Class" not in params_dict.keys():
            # print("Error: Missing Class identifier!")
            # return
            raise ClassMismatch("Error: Missing Class identifier!")
        elif not isinstance(self, globals()[params_dict["Class"]]):
            raise ClassMismatch("Error: Wrong parameter.json for parameter Class!")
        else:
            params_dict.pop("Class", None)
            for key, item in params_dict.items():
                # if isinstance(item, list):
                if hasattr(self, key):
                    if key == "boundaries":
                        self.boundaries.load(item)
                    else:
                        setattr(self, key, item)
                else:
                    print(
                        f"Warning: There is no {key} in the selected Parameter set! {key} is skipped."
                    )

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
            # Custom Encoder

            if attr == "boundaries":
                value = getattr(self, attr).save()
            elif isinstance(getattr(self, attr), np.ndarray):
                value = getattr(self, attr).squeeze().tolist()
            elif isinstance(getattr(self, attr), Path):
                value = getattr(self, attr).__str__()
            else:
                value = getattr(self, attr)
            data_dict[attr] = value
        if not file_path.exists():
            with file_path.open("w") as file:
                file.write("")
        with file_path.open("w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        print(f"Parameters saved to {file_path}")


class NNLSbaseParams(Parameters):
    """Basic NNLS Parameter class. Parent function for both NNLS and NNLSCV."""

    def __init__(
        self,
        params_json: str | Path | None = None,
    ):
        self.reg_order = None
        self.boundaries = self.NNLSBoundaries()
        super().__init__(params_json)
        self.fit_function = Model.NNLS.fit
        self.fit_model = Model.NNLS.model

    class NNLSBoundaries(Boundaries):
        def __init__(self):
            self.scaling = None
            self.dict = dict()
            super().__init__()

        def load(self, data: dict):
            """
            The dictionaries need to be shape according to the following shape:
            "boundaries": {
                "d_range": [],
                "n_bins": []
            }
            Parameters are read starting with the first key descending to bottom level followed by the next key.
            """
            self.dict = data
            self.number_points = data["n_bins"]

        @property
        def parameter_names(self) -> list | None:
            """Returns parameter names from json for IVIM (and generic names vor NNLS)"""
            names = [f"X{i}" for i in range(0, 10)]
            return names

        @parameter_names.setter
        def parameter_names(self, data: dict):
            pass

        @property
        def scaling(self):
            return self._scaling

        @scaling.setter
        def scaling(self, value):
            self._scaling = value

        def apply_scaling(self, value: list) -> list:
            return value

        def get_axis_limits(self) -> tuple:
            return self.dict.get("d_range", [0])[0], self.dict.get("d_range", [1])[1]

    @property
    def fit_function(self):
        """Returns partial of methods corresponding fit function."""
        return partial(
            self._fit_function,
            basis=self.get_basis(),
            max_iter=self.max_iter,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        """Sets fit function."""
        self._fit_function = method

    @property
    def fit_model(self):
        return self._fit_model

    @fit_model.setter
    def fit_model(self, method: Callable):
        self._fit_model = method

    def get_basis(self) -> np.ndarray:
        """Calculates the basis matrix for a given set of b-values."""
        basis = np.exp(
            -np.kron(
                self.b_values,
                self.get_bins(),
            )
        )
        return basis

    def eval_fitting_results(self, results, seg: NiiSeg) -> Results:
        """
        Determines results for the diffusion parameters d & f out of the fitted spectrum.

        Parameters
        ----------
            results
                Pass the results of the fitting process to this function
            seg: NiiSeg
                Get the shape of the spectrum array
        """
        # Create output array for spectrum
        spectrum_shape = np.array(seg.array.shape)
        spectrum_shape[3] = self.get_basis().shape[1]

        fit_results = Results()
        fit_results.spectrum = np.zeros(spectrum_shape)

        bins = self.get_bins()
        for element in results:
            fit_results.spectrum[element[0]] = element[1]

            # Find peaks and calculate fractions
            idx, properties = signal.find_peaks(element[1], height=0.1)
            f_values = properties["peak_heights"]

            # calculate area under the curve fractions by assuming gaussian curve
            if self.reg_order:
                f_fwhms = signal.peak_widths(element[1], idx, rel_height=0.5)[0]
                f_reg = list()
                for peak, fwhm in zip(f_values, f_fwhms):
                    f_reg.append(
                        np.multiply(peak, fwhm)
                        / (2 * math.sqrt(2 * math.log(2)))
                        * math.sqrt(2 * math.pi)
                    )
                f_values = f_reg

            # Save results and normalise f
            fit_results.d[element[0]] = bins[idx]
            fit_results.f[element[0]] = np.divide(f_values, sum(f_values))

            # Set decay curve
            fit_results.curve[element[0]] = self.fit_model(
                self.b_values,
                element[1],
                bins,
            )

        return fit_results

    def apply_AUC_to_results(self, fit_results) -> (dict, dict):
        """Takes the fit results and calculates the AUC for each diffusion regime."""

        regime_boundaries = [0.003, 0.05, 0.3]  # use d_range instead?
        n_regimes = len(regime_boundaries)  # guarantee constant n_entries for heatmaps
        d_AUC, f_AUC = {}, {}

        # Analyse all elements for application of AUC
        for (key, d_values), (_, f_values) in zip(
            fit_results.d.items(), fit_results.f.items()
        ):
            d_AUC[key] = np.zeros(n_regimes)
            f_AUC[key] = np.zeros(n_regimes)

            for regime_idx, regime_boundary in enumerate(regime_boundaries):
                # Check for peaks inside regime
                peaks_in_regime = d_values < regime_boundary

                if not any(peaks_in_regime):
                    continue

                # Merge all peaks within this regime with weighting
                d_regime = d_values[peaks_in_regime]
                f_regime = f_values[peaks_in_regime]
                d_AUC[key][regime_idx] = np.dot(d_regime, f_regime) / sum(f_regime)
                f_AUC[key][regime_idx] = sum(f_regime)

                # Set remaining peaks for analysis of other regimes
                remaining_peaks = d_values >= regime_boundary
                d_values = d_values[remaining_peaks]
                f_values = f_values[remaining_peaks]

        return d_AUC, f_AUC


class NNLSParams(NNLSbaseParams):
    """NNLS Parameter class for regularised fitting."""

    def __init__(
        self,
        params_json: str | Path | None = None,
    ):
        self.reg_order = None
        self.mu = None
        super().__init__(params_json)

    def get_basis(self) -> np.ndarray:
        """Calculates the basis matrix for a given set of b-values in case of regularisation."""
        basis = super().get_basis()
        n_bins = self.boundaries.dict["n_bins"]

        if self.reg_order == 0:
            # no reg returns vanilla basis
            reg = np.zeros([n_bins, n_bins])
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

    def get_pixel_args(self, img: Nii, seg: NiiSeg, *args):
        """Applies regularisation to image data and subsequently calls parent get_pixel_args method."""
        # Enhance image array for regularisation
        reg = np.zeros(
            (
                np.append(
                    np.array(img.array.shape[0:3]),
                    self.boundaries.dict.get("n_bins", 0),
                )
            )
        )
        img_reg = Nii().from_array(np.concatenate((img.array, reg), axis=3))

        pixel_args = super().get_pixel_args(img_reg, seg)

        return pixel_args

    def get_seg_args(self, img: Nii, seg: NiiSeg, seg_number: int, *args) -> zip:
        """Adds regularisation and calls parent get_seg_args method."""
        mean_signal = seg.get_mean_signal(img.array, seg_number)

        # Enhance image array for regularisation
        reg = np.zeros(self.boundaries.dict.get("n_bins", 0))
        reg_signal = np.concatenate((mean_signal, reg), axis=0)

        return zip([[seg_number]], [reg_signal])


class NNLSCVParams(NNLSbaseParams):
    """NNLS Parameter class for CV-regularised fitting."""

    def __init__(
        self,
        params_json: str | Path | None = None,
    ):
        self.tol = None
        self.reg_order = None
        super().__init__(params_json)
        if hasattr(self, "mu") and getattr(self, "mu") is not None and self.tol is None:
            self.tol = self.mu

        self.fit_function = Model.NNLSCV.fit

    @property
    def fit_function(self):
        """Returns partial of methods corresponding fit function."""
        return partial(
            self._fit_function,
            basis=self.get_basis(),
            max_iter=self.max_iter,
            tol=self.tol,
        )

    @fit_function.setter
    def fit_function(self, method):
        self._fit_function = method

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, tol):
        self._tol = tol

    # @staticmethod
    # def _get_G(basis_CV, H, In, mu, signal_CV):
    #     """Determining lambda function G with cross-validation method."""
    #     fit = Model.NNLS.fit(1, signal_CV, np.concatenate((basis_CV, mu * H)))
    #     # fit = NNLS_reg_fit(basis, H, mu, signal)
    #
    #     # Calculating G with CrossValidation method
    #     G = (
    #         norm(signal_CV - np.matmul(basis_CV, fit)) ** 2
    #         / np.trace(
    #             In
    #             - np.matmul(
    #                 np.matmul(
    #                     basis_CV,
    #                     np.linalg.inv(
    #                         np.matmul(basis_CV.T, basis_CV) + np.matmul(mu * H.T, H)
    #                     ),
    #                 ),
    #                 basis_CV.T,
    #             )
    #         )
    #         ** 2
    #     )
    #     return G
    #
    # def get_basis(self) -> np.ndarray:
    #     """Calculates the reg basis matrix using CV."""
    #
    #     basis = super().get_basis()
    #
    #     # Curvature
    #     n_bins = len(basis[1][:])
    #     H = np.array(
    #         -2 * np.identity(n_bins)
    #         + np.diag(np.ones(n_bins - 1), 1)
    #         + np.diag(np.ones(n_bins - 1), -1)
    #     )
    #
    #     # Identity matrix
    #     In = np.identity(len(self.b_values))
    #
    #     Lambda_left = 0.00001
    #     Lambda_right = 8
    #     midpoint = (Lambda_right + Lambda_left) / 2
    #
    #     # Function (+ delta) and derivative f at left point
    #     G_left = self._get_G(basis, H, In, Lambda_left, signal)
    #     G_leftDiff = self._get_G(basis, H, In, Lambda_left + tol, signal)
    #     f_left = (G_leftDiff - G_left) / tol
    #
    #     count = 0
    #     while abs(Lambda_right - Lambda_left) > tol:
    #         midpoint = (Lambda_right + Lambda_left) / 2
    #         # Function (+ delta) and derivative f at middle point
    #         G_middle = self._get_G(basis, H, In, midpoint, signal)
    #         G_middleDiff = self._get_G(basis, H, In, midpoint + tol, signal)
    #         f_middle = (G_middleDiff - G_middle) / tol
    #
    #         if count > 100:
    #             print("Original choice of mu might not bracket minimum.")
    #             break
    #
    #         # Continue with logic
    #         if f_left * f_middle > 0:
    #             # Throw away left half
    #             Lambda_left = midpoint
    #             f_left = f_middle
    #         else:
    #             # Throw away right half
    #             Lambda_right = midpoint
    #         count = +1
    #     self.mu = midpoint
    #
    #     # Build reg CV basis
    #     # basis = np.matmul(
    #     #     np.concatenate((basis, mu * H)).T, np.concatenate((basis, self.mu * H))
    #     # )
    #
    #     return np.concatenate((basis, mu * H))


class IVIMParams(Parameters):
    """
    Multi-exponential Parameter class used for the IVIM model.

    Child-class methods:
    -------------------
    n_components(int | str)
        Sets number of compartments of current IVIM model.
    set_boundaries()
        Sets lower and upper fitting boundaries and starting values for IVIM.
    """

    def __init__(self, params_json: str | Path | None = None):
        self.boundaries = self.IVIMBoundaries()
        self.n_components = None
        self.TM = None
        super().__init__(params_json)
        self.fit_function = Model.IVIM.fit
        self.fit_model = Model.IVIM.wrapper

    class IVIMBoundaries(Boundaries):
        def __init__(self):
            self.dict = None
            super().__init__()

        @property
        def parameter_names(self) -> list | None:
            """Returns parameter names from json for IVIM (and generic names vor NNLS)"""
            names = list()
            for key in self.dict:
                for subkey in self.dict[key]:
                    names.append(key + "_" + subkey)
            names = self.apply_scaling(names)
            if len(names) == 0:
                names = None
            return names
            # names = dict()
            # for key in self.dict():
            #     for subkey in self.dict[key]:
            #         names[key] = subkey
            # return names

        # @parameter_names.setter
        # def parameter_names(self, data: dict):
        #     self.dict = data

        @property
        def scaling(self):
            return self._scaling

        @scaling.setter
        def scaling(self, value):
            self._scaling = value

        @property
        def start_values(self):
            return self._get_boundary(0)

        @start_values.setter
        def start_values(self, x0: list | np.ndarray):
            self._set_boundary(0, x0)

        @property
        def lower_stop_values(self):
            return self._get_boundary(1)

        @lower_stop_values.setter
        def lower_stop_values(self, lb: list | np.ndarray):
            self._set_boundary(1, lb)

        @property
        def upper_stop_values(self):
            return self._get_boundary(2)

        @upper_stop_values.setter
        def upper_stop_values(self, ub: list | np.ndarray):
            self._set_boundary(2, ub)

        def load(self, data: dict):
            """
            "D": {
                "<NAME>": [x0, lb, ub],
                ...
            },
            "f": {
                "<NAME>": [x0, lb, ub],
                ...
            }
            "S": {
                "<NAME>": [x0, lb, ub],
            }
            """
            self.dict = data.copy()

        def save(self) -> dict:
            _dict = super().save()
            return _dict

        def apply_scaling(self, value: list) -> list:
            if isinstance(self._scaling, str):
                if self.scaling == "S/S0" and "S" in self.dict.keys():
                    # with S/S0 the number of Parameters is reduced.
                    value = value[:-1]
            elif isinstance(self.scaling, (int, float)):
                pass
            return value

        def _get_boundary(self, pos: int):
            # TODO: Remove scale when the fitting dlg is reworked accordingly, adjust dlg accordingly
            values = list()
            for key in self.dict:
                for subkey in self.dict[key]:
                    values.append(self.dict[key][subkey][pos])
            values = self.apply_scaling(values)
            values = np.array(values)
            return values

        def _set_boundary(self, pos: int, values: list | np.ndarray):
            idx = 0
            for key in self.dict:
                for subkey in self.dict[key]:
                    self.dict[key][subkey][pos] = values[idx]
                    idx += 1

        def get_axis_limits(self) -> tuple:
            _min = min(self.lower_stop_values)
            _max = max(self.upper_stop_values)
            return _min, _max

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value: int | str | None):
        if isinstance(value, str):
            if "MonoExp" in value:
                value = 1
            elif "BiExp" in value:
                value = 2
            elif "TriExp" in value:
                value = 3
        self._n_components = value
        # if self.boundaries["x0"] is None or not len(self.boundaries["x0"]) == value:
        #     self.set_boundaries()

    @property
    def fit_function(self):
        # Integrity Check necessary
        return partial(
            self._fit_function,
            b_values=self.get_basis(),
            x0=self.boundaries.start_values,
            lb=self.boundaries.lower_stop_values,
            ub=self.boundaries.upper_stop_values,
            n_components=self.n_components,
            max_iter=self.max_iter,
            TM=self.TM,
            scale_image=self.scale_image if isinstance(self.scale_image, str) else None,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        """Sets fit function."""
        self._fit_function = method

    @property
    def fit_model(self):
        return self._fit_model(
            n_components=self.n_components,
            TM=self.TM,
            scale_image=self.scale_image if isinstance(self.scale_image, str) else None,
        )

    @fit_model.setter
    def fit_model(self, method: Callable):
        """Sets fitting model."""
        self._fit_model = method

    def load_from_json(self, params_json: str | Path | None = None):
        super().load_from_json(params_json)
        # keys = ["x0", "lb", "ub"]
        # for key in keys:
        #     if not isinstance(self.boundaries[key], np.ndarray):
        #         self.boundaries[key] = np.array(self.boundaries[key])

    def get_basis(self):
        """Calculates the basis matrix for a given set of b-values."""
        return np.squeeze(self.b_values)

    def eval_fitting_results(self, results, seg) -> Results:
        """
        Assigns fit results to the diffusion parameters d & f.

        Parameters
        ----------
            results
                Pass the results of the fitting process to this function
            seg: NiiSeg
                Get the shape of the spectrum array
        """
        # prepare arrays
        fit_results = Results()
        for element in results:
            fit_results.raw[element[0]] = element[1]
            fit_results.S0[element[0]] = element[1][-1]
            fit_results.d[element[0]] = element[1][0 : self.n_components]
            f_new = np.zeros(self.n_components)

            if isinstance(self.scale_image, str) and self.scale_image == "S/S0":
                f_new[: self.n_components - 1] = element[1][self.n_components :]
                if np.sum(element[1][self.n_components :]) > 1:
                    f_new = np.zeros(self.n_components)
                    print(f"Fit error for Pixel {element[0]}")
                else:
                    f_new[-1] = 1 - np.sum(element[1][self.n_components :])
            else:
                f_new[: self.n_components - 1] = element[1][self.n_components : -1]
                if np.sum(element[1][self.n_components : -1]) > 1:
                    f_new = np.zeros(self.n_components)
                    print(f"Fit error for Pixel {element[0]}")
                else:
                    f_new[-1] = 1 - np.sum(element[1][self.n_components : -1])

            fit_results.f[element[0]] = f_new

            # add curve fit
            fit_results.curve[element[0]] = self.fit_model(self.b_values, *element[1])

            # add additional T1 results if necessary
            if self.TM:
                fit_results.T1[element[0]] = [element[1][2]]

        fit_results = self.set_spectrum_from_variables(fit_results, seg)

        return fit_results

    def set_spectrum_from_variables(
        self, fit_results: Results, seg: NiiSeg, number_points: int = 250
    ):
        # adjust d-values according to bins/d-values
        """
        Creates a spectrum out of the distinct IVIM results to enable comparison to NNLS results.

        The set_spectrum_from_variables function takes the fit_results and seg objects as input. It then creates a
        new spectrum array with the same shape as the seg object, but with an additional dimension for each bin.
        The function then loops through all pixels in fit_results and adds up all contributions to each bin from
        every fiber component at that pixel position. Finally, it returns this new spectrum.

        Parameters
        ----------
            fit_results: Results
                Store the results of the fit
            seg: NiiSeg
                Get the shape of the segmentation file
            number_points: int
                The number of points used for the spectrum
        """
        d_values = self.get_bins()

        # Prepare spectrum for dyn
        # TODO: Rework to set proper signals using a set number of points (not included in fit json)
        new_shape = np.array(seg.array.shape)
        new_shape[3] = number_points
        spectrum = np.zeros(new_shape)

        for pixel_pos in fit_results.d:
            temp_spec = np.zeros(number_points)
            d_new = list()
            for D, F in zip(fit_results.d[pixel_pos], fit_results.f[pixel_pos]):
                index = np.unravel_index(
                    np.argmin(abs(d_values - D), axis=None),
                    d_values.shape,
                )[0].astype(int)
                d_new.append(d_values[index])
                temp_spec = temp_spec + F * signal.unit_impulse(number_points, index)
                spectrum[pixel_pos] = temp_spec
        fit_results.spectrum = spectrum
        return fit_results

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """Performs S/S0 normalization on an array"""
        img_new = np.zeros(img.shape)
        for i, j, k in zip(*np.nonzero(img[:, :, :, 0])):
            img_new[i, j, k, :] = img[i, j, k, :] / img[i, j, k, 0]
        return img_new


class IDEALParams(IVIMParams):
    def __init__(
        self,
        params_json: Path | str = None,
    ):
        """
        IDEAL fitting Parameter class.

        Attributes
        ----------
        params_json: Parameter json file containing basic fitting parameters.

        """
        self.tolerance = None
        self.dimension_steps = None
        self.segmentation_threshold = None
        super().__init__(params_json)
        self.fit_function = Model.IVIM.fit
        self.fit_model = Model.IVIM.wrapper

    @property
    def fit_function(self):
        return partial(
            self._fit_function,
            b_values=self.get_basis(),
            n_components=self.n_components,
            max_iter=self.max_iter,
            TM=None,
            scale_image=self.scale_image if isinstance(self.scale_image, str) else None,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        self._fit_function = method

    @property
    def fit_model(self):
        return self._fit_model(
            n_components=self.n_components,
            TM=self.TM,
            scale_image=self.scale_image if isinstance(self.scale_image, str) else None,
        )

    @fit_model.setter
    def fit_model(self, method: Callable):
        self._fit_model = method

    @property
    def dimension_steps(self):
        return self._dimension_steps

    @dimension_steps.setter
    def dimension_steps(self, value):
        if isinstance(value, list):
            steps = np.array(value)
        elif isinstance(value, np.ndarray):
            steps = value
        elif value is None:
            # TODO: Special None Type handling? (IDEAL)
            steps = None
        else:
            raise TypeError()
        # Sort Dimension steps
        self._dimension_steps = (
            np.array(sorted(steps, key=lambda x: x[1], reverse=True))
            if steps is not None
            else None
        )

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: list | np.ndarray | None):
        """
        Tolerance for IDEAL step boundaries in relative values.

        value: list | np.ndarray
            All stored values need to be floats with 0 < value < 1
        """
        if isinstance(value, list):
            self._tolerance = np.array(value)
        elif isinstance(value, np.ndarray):
            self._tolerance = value
        elif value is None:
            self._tolerance = None
        else:
            raise TypeError()

    @property
    def segmentation_threshold(self):
        return self._segment_threshold

    @segmentation_threshold.setter
    def segmentation_threshold(self, value: float | None):
        if isinstance(value, float):
            self._segment_threshold = value
        elif value is None:
            self._segment_threshold = 0.025
        else:
            raise TypeError()

    def get_basis(self):
        return np.squeeze(self.b_values)

    def get_pixel_args(self, img: np.ndarray, seg: np.ndarray, *args) -> zip:
        # Behaves the same way as the original parent funktion with the difference that instead of Nii objects
        # np.ndarrays are passed. Also needs to pack all additional fitting parameters [x0, lb, ub]
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
            (img[i, j, k, :] for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
            (
                args[0][i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
            (
                args[1][i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
            (
                args[2][i, j, k, :]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
        )
        return pixel_args

    def interpolate_start_values_2d(
        self, boundary: np.ndarray, matrix_shape: np.ndarray, n_pools: int | None = None
    ) -> np.ndarray:
        """
        Interpolate starting values for the given boundary.

        boundary: np.ndarray of shape(x, y, z, n_variables).
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        # if boundary.shape[0:1] < matrix_shape:
        boundary_new = np.zeros(
            (matrix_shape[0], matrix_shape[1], boundary.shape[2], boundary.shape[3])
        )
        arg_list = zip(
            ([i, j] for i, j in zip(*np.nonzero(np.ones(boundary.shape[2:4])))),
            (
                boundary[:, :, i, j]
                for i, j in zip(*np.nonzero(np.ones(boundary.shape[2:4])))
            ),
        )
        func = partial(self.interpolate_array_multithreading, matrix_shape=matrix_shape)
        results = multithreader(func, arg_list, n_pools=n_pools)
        return sort_interpolated_array(results, array=boundary_new)

    def interpolate_img(
        self,
        img: np.ndarray,
        matrix_shape: np.ndarray | list | tuple,
        n_pools: int | None = None,
    ) -> np.ndarray:
        """
        Interpolate image to desired size in 2D.

        img: np.ndarray of shape(x, y, z, n_bvalues)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        # get new empty image
        img_new = np.zeros(
            (matrix_shape[0], matrix_shape[1], img.shape[2], img.shape[3])
        )
        # get x*y image planes for all slices and decay points
        arg_list = zip(
            ([i, j] for i, j in zip(*np.nonzero(np.ones(img.shape[2:4])))),
            (img[:, :, i, j] for i, j in zip(*np.nonzero(np.ones(img.shape[2:4])))),
        )
        func = partial(
            self.interpolate_array_multithreading,
            matrix_shape=matrix_shape,
        )
        results = multithreader(func, arg_list, n_pools=n_pools)
        return sort_interpolated_array(results, array=img_new)

    def interpolate_seg(
        self,
        seg: np.ndarray,
        matrix_shape: np.ndarray | list | tuple,
        threshold: float,
        n_pools: int | None = 4,
    ) -> np.ndarray:
        """
        Interpolate segmentation to desired size in 2D and apply threshold.

        seg: np.ndarray of shape(x, y, z)
        matrix_shape: np.ndarray of shape(2, 1) containing new in plane matrix size
        """
        seg_new = np.zeros((matrix_shape[0], matrix_shape[1], seg.shape[2], 1))

        # get x*y image planes for all slices and decay points
        arg_list = zip(
            ([i, j] for i, j in zip(*np.nonzero(np.ones(seg.shape[2:4])))),
            (seg[:, :, i, j] for i, j in zip(*np.nonzero(np.ones(seg.shape[2:4])))),
        )
        func = partial(
            self.interpolate_array_multithreading,
            matrix_shape=matrix_shape,
        )
        results = multithreader(func, arg_list, n_pools=n_pools)
        seg_new = sort_interpolated_array(results, seg_new)

        # Make sure Segmentation is binary
        seg_new[seg_new < threshold] = 0
        seg_new[seg_new > threshold] = 1

        # Check seg size. Needs to be M x N x Z x 1
        while len(seg_new.shape) < 4:
            seg_new = np.expand_dims(seg_new, axis=len(seg_new.shape))
        return seg_new

    @staticmethod
    def interpolate_array_multithreading(
        idx: tuple | list, array: np.ndarray, matrix_shape: np.ndarray
    ):
        # Cv-less version of interpolate_image
        # def interpolate_array(arr: np.ndarray, shape: np.ndarray):
        #     """Interpolate 2D array to new shape."""
        #
        #     x, y = np.meshgrid(
        #         np.linspace(0, 1, arr.shape[1]), np.linspace(0, 1, arr.shape[0])
        #     )
        #     x_new, y_new = np.meshgrid(
        #         np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0])
        #     )
        #     points = np.column_stack((x.flatten(), y.flatten()))
        #     values = arr.flatten()
        #     new_values = griddata(points, values, (x_new, y_new), method="cubic")
        #     return np.reshape(new_values, shape)

        def interpolate_array_cv(arr: np.ndarray, shape: np.ndarray):
            return cv2.resize(arr, shape, interpolation=cv2.INTER_CUBIC)

        # def interpolate_array_scipy

        array = interpolate_array_cv(array, matrix_shape)
        return idx, array

    def eval_fitting_results(self, results: np.ndarray, seg: NiiSeg) -> Results:
        """
        Evaluate fitting results for the IDEAL method.

        Parameters
        ----------
            results
                Pass the results of the fitting process to this function
            seg: NiiSeg
                Get the shape of the spectrum array
        """
        coordinates = seg.get_seg_indices("nonzero")
        # results_zip = list(zip(coordinates, results[coordinates]))
        results_zip = zip(
            (coord for coord in coordinates), (results[coord] for coord in coordinates)
        )
        fit_results = super().eval_fitting_results(results_zip, seg)
        return fit_results


class JsonImporter:
    def __init__(self, json_file: Path | str):
        self.json_file = json_file
        self.parameters = None

    def load_json(self):
        if self.json_file.is_file():
            with open(self.json_file, "r") as f:
                params_dict = json.load(f)

        if "Class" not in params_dict.keys():
            raise ClassMismatch("Error: Missing Class identifier!")
        elif not globals().get(params_dict["Class"], False):
            raise ClassMismatch("Error: Wrong parameter.json for parameter Class!")
        else:
            self.parameters = globals()[params_dict["Class"]]()
            self.parameters.json = Path(self.json_file).resolve()
            params_dict.pop("Class", None)
            for key, item in params_dict.items():
                # if isinstance(item, list):
                if hasattr(self.parameters, key):
                    setattr(self.parameters, key, item)
                else:
                    print(
                        f"Warning: There is no {key} in the selected Parameter set! {key} is skipped."
                    )
            if isinstance(self.parameters, IVIMParams):
                keys = ["x0", "lb", "ub"]
                for key in keys:
                    if not isinstance(self.parameters.boundaries[key], np.ndarray):
                        self.parameters.boundaries[key] = np.array(
                            self.parameters.boundaries[key]
                        )
            return self.parameters
