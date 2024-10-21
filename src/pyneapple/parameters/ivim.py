from __future__ import annotations

import numpy as np
from pathlib import Path
from functools import partial
from typing import Callable
from scipy import signal

from .parameters import Parameters
from .boundaries import IVIMBoundaries
from ..models import IVIM, IVIMFixedComponent
from ..results.results import CustomDict


class IVIMParams(Parameters):
    """
    Multi-exponential Parameter class used for the IVIM model.

    Attributes:
        boundaries (IVIMBoundaries): Boundaries for IVIM model.
        TM (bool): Flag for T1 mapping.

    Methods:
        set_boundaries(): Sets lower and upper fitting boundaries and starting values
            for IVIM.
    """

    def __init__(self, params_json: str | Path | None = None):
        self.boundaries = IVIMBoundaries()
        self.n_components = None
        self.TM = None
        super().__init__(params_json)
        self.fit_function = IVIM.fit
        self.fit_model = IVIM.wrapper

    @property
    def n_components(self):
        """Number of components for the IVIM model."""
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
    def fit_function(self) -> partial:
        """Returns the fit function partially initialized."""
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
    def fit_model(self) -> Callable:
        """Return fit model with set parameters."""
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
        """Load parameter information from json file."""
        super().load_from_json(params_json)

    def get_basis(self) -> np.ndarray:
        """Calculates the basis matrix for a given set of b-values."""
        return np.squeeze(self.b_values)

    # ------ The following methods are related to evaluating the fitting results

    def eval_fitting_results(self, results, **kwargs) -> dict:
        """
        Assigns fit results to the diffusion parameters d & f.

        Args:
            results (list): Pass the results of the fitting process to this function
            seg (NiiSeg): Get the shape of the spectrum array

        Returns
            fitted_results (dict): The results of the fitting process combined in a
                dictionary. Each entry holds a dictionary containing the different
                results.
        """
        # prepare arrays
        raw, S0, d, f, curve, t_one = dict(), dict(), dict(), dict(), dict(), dict()

        for element in results:
            S0[element[0]] = self.get_s_0_values_from_results(element[1])
            f[element[0]] = self.get_fractions_from_results(element[1])
            d[element[0]] = self.get_diffusion_values_from_results(element[1])
            t_one[element[0]] = self.get_t_one_from_results(element[1])
            raw[element[0]] = self.get_raw_result(
                d[element[0]],
                f[element[0]],
                S0[element[0]],
                t_one[element[0]],
            )
            curve[element[0]] = self.fit_model(
                self.b_values,
                *d[element[0]],
                *f[element[0]],
                S0[element[0]],
                t_one[element[0]],
            )

        spectrum = self.set_spectrum_from_variables(d, f)

        fit_results = {
            "raw": raw,
            "S0": S0,
            "d": d,
            "f": f,
            "curve": curve,
            "spectrum": spectrum,
        }
        if self.TM:
            fit_results.update({"T1": t_one})

        return fit_results

    def get_fractions_from_results(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the fractions of the diffusion components.

        Args:
            results (np.ndarray): Results of the fitting process.
            kwargs (dict):
                n_components (int): set the number of diffusion components manually.
        Returns:
            f_new (np.ndarray): Fractions of the diffusion components.
        """

        n_components = kwargs.get("n_components", self.n_components)
        f_new = np.zeros(self.n_components)
        if isinstance(self.scale_image, str) and self.scale_image == "S/S0":
            # for S/S0 one parameter less is fitted
            f_new[:n_components] = results[n_components:]
        else:
            if n_components > 1:
                f_new[: n_components - 1] = results[
                    n_components : (2 * n_components - 1)
                ]
            else:
                f_new[0] = 1
        if np.sum(f_new) > 1:  # fit error
            f_new = np.zeros(n_components)
        else:
            f_new[-1] = 1 - np.sum(f_new)
        return f_new

    def get_diffusion_values_from_results(
        self, results: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Extract diffusion values from the results list."""
        return results[: self.n_components]

    def get_s_0_values_from_results(self, results: np.ndarray) -> np.ndarray:
        """Extract S0 values from the results list."""
        if isinstance(self.scale_image, str) and self.scale_image == "S/S0":
            return 1
        else:
            if self.TM:
                return results[-2]
            else:
                return results[-1]

    def get_t_one_from_results(self, results: np.ndarray) -> np.ndarray | None:
        """Extract T1 values from the results list."""
        if self.TM:
            return results[-1]
        else:
            return None

    def get_raw_result(
        self,
        d: np.ndarray | list,
        f: np.ndarray | list,
        s_0: np.ndarray | list,
        t_1: np.ndarray | list,
        **kwargs,
    ) -> list:
        """
        Returns the raw results of the fitting process.

        Args:
            d (np.ndarray): containing the diffusion values
            f (np.ndarray): containing the fractions
            s_0 (np.ndarray): containing the S0 values
            t_1 (np.ndarray): containing the T1 values

        Returns:
            raw (list): All raw fitting results in form of a list like they are returned
                by the fitting process.
        """
        raw = [*d, *f]
        if not isinstance(self.scale_image, str) and self.scale_image == "S/S0":
            raw.append(s_0)
        if self.TM:
            raw.append(t_1)
        return raw

    def set_spectrum_from_variables(
        self, d: dict, f: dict, number_points: int | None = None
    ) -> CustomDict:
        # adjust d-values according to bins/d-values
        """
        Creates a spectrum out of the distinct IVIM results to enable comparison to NNLS
        results.

        The set_spectrum_from_variables function takes the fit_results and seg objects as
        input. It then creates a new spectrum array with the same shape as the seg
        object, but with an additional dimension for each bin. The function then loops
        through all pixels in fit_results and adds up all contributions to each bin from
        every fiber component at that pixel position. Finally, it returns this new
        spectrum.

        Args:
            d (dict): Dictionary containing diffusion values
            f (dict): Dictionary containing diffusion fractions
            number_points (int): The number of points used for the spectrum
        Returns:
            spectrum (CustomDict): The spectrum array with the same shape as the seg
                object.
        """
        d_values = self.get_bins()
        spectrum = CustomDict()

        if number_points is None:
            number_points = self.boundaries.number_points

        for pixel_pos in d:
            temp_spec = np.zeros(number_points)
            d_new = list()
            # TODO: As for now the d peaks are plotted on discrete values given by the number of points and
            # TODO: the fitting interval. This should be changed to the actual values.
            for D, F in zip(d[pixel_pos], f[pixel_pos]):
                index = np.unravel_index(
                    np.argmin(abs(d_values - D), axis=None),
                    d_values.shape,
                )[0].astype(int)
                d_new.append(d_values[index])
                temp_spec = temp_spec + F * signal.unit_impulse(number_points, index)
                spectrum[pixel_pos] = temp_spec
        return spectrum

    # ------

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """Performs S/S0 normalization on an array"""
        img_new = np.zeros(img.shape)
        for i, j, k in zip(*np.nonzero(img[:, :, :, 0])):
            img_new[i, j, k, :] = img[i, j, k, :] / img[i, j, k, 0]
        return img_new


class IVIMSegmentedParams(IVIMParams):
    """IVIM based parameters for segmented fitting fixing one component.

    Attributes:
        params_fixed (IVIMParams): Fixed parameters for segmented fitting.
        options (dict): Options for segmented fitting.
    Methods:
        set_options: Setting necessary options for segmented IVIM fitting.
        get_fixed_fit_results: Extract the calculated fixed values per pixel from
            results.
        get_pixel_args: Returns the pixel arguments needed for the second fitting step.
        get_pixel_args_fixed: Works the same way as the IVIMParams version but can take
            reduced b_values into account.
        eval_fitting_results: Assigns fit results to the diffusion parameters d & f.
        get_diffusion_values_from_results: Returns the diffusion values from the results
            and adds the fixed component to the results.
        get_fractions_from_results: Returns the fractions of the diffusion components
            with adjustments for segmented fitting.
    """

    def __init__(
        self,
        params_json: str | Path | None = None,
        **options,
    ):
        """
        Multi-exponential Parameter class used for the segmented IVIM fitting.

        Args:
            params_json (str | Path): Json containing fitting parameters
            options (**kwargs):  options for segmented fitting
                fixed_component (str):  In the shape of "D_slow" with "_" as separator
                    for dict
                fixed_t1 (bool):  Set T1 for pre fitting
                reduced_b_values (list): of b_values used for first fitting (second
                    fitting is always performed with all)

        """
        super().__init__(params_json)
        # Set mono / ADC default params set as starting point
        self.options = options
        self.params_fixed = IVIMParams()
        self.init_fixed_params()
        self.fit_model = IVIM.wrapper
        self.fit_function = IVIMFixedComponent.fit
        # change parameters according to selected
        self.set_options(
            options.get("fixed_component", None),
            options.get("fixed_t1", False),
            options.get("reduced_b_values", None),
        )

    @property
    def fit_function(self):
        """Returns the fit function partially initialized."""
        return partial(
            self._fit_function,
            b_values=self.get_basis(),
            x0=self.boundaries.start_values,
            lb=self.boundaries.lower_stop_values,
            ub=self.boundaries.upper_stop_values,
            n_components=self.n_components,
            max_iter=self.max_iter,
            TM=self.TM if not self.options["fixed_t1"] else None,
            scale_image=self.scale_image if isinstance(self.scale_image, str) else None,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        """Sets fit function."""
        self._fit_function = method

    def init_fixed_params(self):
        self.params_fixed.n_components = 1
        self.params_fixed.max_iter = self.max_iter
        self.params_fixed.n_pools = self.n_pools
        self.params_fixed.fit_area = self.fit_area
        self.params_fixed.scale_image = self.scale_image

    def set_options(
        self,
        fixed_component: str | None,
        fixed_t1: bool,
        reduced_b_values: list | None = None,
    ) -> None:
        """Setting necessary options for segmented IVIM fitting.

        Args:
            fixed_component (str): in the shape of "D_slow" with "_" as separator for
                dict
            fixed_t1 (bool): set T1 for pre fitting
            reduced_b_values (list): of b_values used for first fitting (second fitting
                is always performed with all)
        """

        # store options
        self.options["fixed_component"] = fixed_component
        self.options["fixed_t1"] = fixed_t1
        self.options["reduced_b_values"] = reduced_b_values

        # if t1 pre fitting is wanted TM needs to be deployed and flag set accordingly
        if fixed_t1:
            self.params_fixed.TM = self.TM
            self.TM = None

        if fixed_component:
            # Prepare Boundaries for the first fit
            dict_keys = fixed_component.split("_")
            boundary_dict = dict()
            boundary_dict[dict_keys[0]] = {}
            boundary_dict[dict_keys[0]][dict_keys[1]] = self.boundaries.dict[
                dict_keys[0]
            ][dict_keys[1]]

            # Add T1 boundaries if needed
            if fixed_t1:
                boundary_dict["T"] = self.boundaries.dict.pop(
                    "T", KeyError("T has no defined boundaries.")
                )

            self.params_fixed.scale_image = self.scale_image
            # If S0 should be fitted the parameter should be passed to the fixed parameters class
            if not isinstance(self.scale_image, str) and not self.scale_image == "S/S0":
                boundary_dict["S"] = {}
                boundary_dict["S"]["0"] = self.boundaries.dict["S"]["0"]
            # load dict
            self.params_fixed.boundaries.load(boundary_dict)

            # Prepare Boundaries for the second fit
            # Remove unused Parameter
            boundary_dict = self.boundaries.dict.copy()
            boundary_dict[dict_keys[0]].pop(dict_keys[1])

            # Load dict
            self.boundaries.load(boundary_dict)

        if reduced_b_values:
            self.params_fixed.b_values = reduced_b_values
        else:
            self.params_fixed.b_values = self.b_values

    def get_fixed_fit_results(self, results: list | tuple) -> list:
        """Extract the calculated fixed values per pixel from results.

        Args:
            results (list): of tuples containing the results of the fitting process
                [0]: tuple containing pixel coordinates
                [1]: list containing the fitting results

        Returns:
            [d, t1] (list): containing the fixed values for the diffusion constant and
                T1 values
        """

        d = dict()
        if self.options["fixed_t1"]:
            t1 = dict()

        for element in results:
            d[element[0]] = element[1][0]
            if self.options["fixed_t1"]:
                t1[element[0]] = element[1][1]

        return [d, t1] if self.options["fixed_t1"] else [d]

    def get_pixel_args(self, img: np.ndarray, seg: np.ndarray, *fixed_results) -> zip:
        """Returns the pixel arguments needed for the second fitting step.

        For each pixel the coordinates (x,y,z), the corresponding pixel decay signal and the pre-fitted
        decay constant (and T1 value) are packed.

        Args:
            img (Nii): Nifti image
            seg (NiiSeg): Segmentation image
            fixed_results (list): containing np.arrays of seg.shape

        Returns:
            pixel_args (zip): containing the pixel arguments for the fitting process
        """

        indexes = [(i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))]
        signals = [
            img[i, j, k] for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
        ]
        adc_s = [
            fixed_results[0][i, j, k]
            for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
        ]

        if self.options["fixed_t1"]:
            t_ones = [
                fixed_results[1][i, j, k]
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ]
            return zip(indexes, signals, adc_s, t_ones)
        else:
            return zip(indexes, signals, adc_s)

    def get_pixel_args_fixed(self, img: np.ndarray, seg: np.ndarray, *args) -> zip:
        """Works the same way as the IVIMParams version but can take reduced b_values
            into account.

        Args:
            img (np.ndarray): Nifti image
            seg (np.ndarray): Segmentation image
            args (list): containing np.arrays of seg.shape

        Returns:
            pixel_args (zip): containing the pixel arguments for the fitting process
        """
        if not self.options["reduced_b_values"]:
            pixel_args = super().get_pixel_args(img, seg, args)
        else:
            # get b_value positions
            indexes = np.where(
                np.isin(
                    self.b_values,
                    self.options["reduced_b_values"],
                )
            )[0]
            img_reduced = img[:, :, :, indexes]
            pixel_args = zip(
                ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
                (
                    img_reduced[i, j, k, :]
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
                ),
            )

        return pixel_args

    def eval_fitting_results(self, results: list, **kwargs) -> dict:
        """
        Assigns fit results to the diffusion parameters d & f.

        Args:
            results (list): of tuples containing the fitting results
                [0]: tuple containing pixel coordinates
                [1]: list containing the fitting
            **kwargs: additional options
                fixed_component: list(dict, dict)
                    Dictionary holding results from first fitting step

        Returns:
            fitted_results (dict): The results of the fitting process combined in a
                dictionary. Each entry holds a dictionary containing the different
                results.
        """

        fixed_component = kwargs.get("fixed_component", [[None]])

        raw, S0, d, f, curve, t_one = dict(), dict(), dict(), dict(), dict(), dict()

        for element in results:
            S0[element[0]] = self.get_s_0_values_from_results(element[1])
            f[element[0]] = self.get_fractions_from_results(
                element[1]  # , n_components=self.n_components - 1
            )
            d[element[0]] = self.get_diffusion_values_from_results(
                element[1], fixed_component=fixed_component[0][element[0]]
            )
            t_one[element[0]] = [element[1][1]]
            raw[element[0]] = self.get_raw_result(
                d[element[0]],
                f[element[0]],
                S0[element[0]],
                t_one[element[0]],
                fixed_component=fixed_component[0][element[0]],
            )
            curve[element[0]] = self.fit_model(
                self.b_values,
                *d[element[0]],
                *f[element[0]][:-1],
                S0[element[0]],
                t_one[element[0]],
            )

        spectrum = self.set_spectrum_from_variables(d, f)

        fit_results = {
            "raw": raw,
            "S0": S0,
            "d": d,
            "f": f,
            "curve": curve,
            "spectrum": spectrum,
        }
        if self.TM:
            fit_results.update({"T1": t_one})

        return fit_results

    def get_diffusion_values_from_results(self, results: np.ndarray, **kwargs):
        """Returns the diffusion values from the results and adds the fixed component to the results.

        Args:
            results (np.ndarray): containing the fitting results
            **kwargs:
                fixed_component (list | np.ndarray): containing the fixed component
                    results

        Returns:
            d_new (np.ndarray): containing the diffusion values
        """
        fixed_component = kwargs.get("fixed_component", None)

        d_new = np.zeros(self.n_components)
        # add fixed component
        d_new[0] = fixed_component
        # since D_slow aka ADC is the default fitting parameter it is always at 0
        # this will cause issues if the fixed component is not the first component
        d_new[1:] = results[: self.n_components - 1]
        return d_new

    def get_fractions_from_results(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the fractions of the diffusion components with adjustments for segmented fitting.

        Args:
            results: np.ndarray
                Results of the fitting process.
            **kwargs:
                n_components: int set the number of diffusion components manually.
        """

        n_components = kwargs.get("n_components", self.n_components)
        f_new = np.zeros(self.n_components)
        if isinstance(self.scale_image, str) and self.scale_image == "S/S0":
            # for S/S0 one parameter less is fitted
            f_new[:n_components] = results[n_components:]
        else:
            if n_components > 1:
                f_new[: n_components - 1] = results[
                    (n_components - 1) : (2 * n_components - 2)
                ]
        if np.sum(f_new) > 1:  # fit error
            f_new = np.zeros(n_components)
        else:
            f_new[-1] = 1 - np.sum(f_new)
        return f_new
