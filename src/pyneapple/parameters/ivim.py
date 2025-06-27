"""IVIM Parameters Module
Contains all necessary classes for the IVIM model fitting.

Classes:
    IVIMParams: Multi-exponential Parameter class used for the IVIM model.
    IVIMSegmentedParams: IVIM based parameters for segmented fitting fixing one
        component.

"""

from __future__ import annotations

from pathlib import Path
from functools import partial
from typing import Callable
import numpy as np

from ..utils.logger import logger
from .parameters import BaseParams
from .boundaries import IVIMBoundaries
from .. import models
from radimgarray import RadImgArray, SegImgArray


class IVIMParams(BaseParams):
    """
    Multi-exponential Parameter class used for the IVIM model.

    Attributes:
        boundaries (IVIMBoundaries): Boundaries for IVIM model.
        mixing_time (bool): Flag for T1 mapping.

    Methods:
        set_boundaries(): Sets lower and upper fitting boundaries and starting values
            for IVIM.
    """

    def __init__(self, params_json: str | Path | None = None):
        self.boundaries = IVIMBoundaries()
        self.n_components = 0
        self.fit_reduced = False
        self.fit_t1 = False
        self.mixing_time = None
        super().__init__(params_json)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: str):
        """Sets the model for the IVIM fitting.

        Note:
            Only exponential models are supported. The model string should be in the json file loaded.
            fit_model and fit_function are set accordingly but will only work for fit_type "single" and "multi".
            For gpu based fitting the model (str) itself is used to get the corresponding ID for the model in pygpufit.
        """
        model_split = model.split("_")
        if not "exp" in model_split[0].lower():
            error_msg = f"Only exponential models are supported. Got: {model}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            self.fit_function = models.fit_curve
            if "mono" in model_split[0].lower():
                self.n_components = 1
                self.fit_model = models.mono_wrapper
            elif "bi" in model_split[0].lower():
                self.n_components = 2
                self.fit_model = models.bi_wrapper
            elif "tri" in model_split[0].lower():
                self.n_components = 3
                self.fit_model = models.tri_wrapper
            else:
                error_msg = f"Only mono-, bi- and tri-exponential models are supported atm. Got: {model_split[0]}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        for string in model_split[1:]:
            if "reduced" in string.lower() or "red" in string.lower():
                self.fit_reduced = True
            elif "t1" in string.lower():
                self.fit_t1 = True
        self._model = model.upper()

    @property
    def fit_model(self) -> Callable:
        """Return fit model with set parameters."""
        return partial(
            self._fit_model(
                reduced=self.fit_reduced,
                mixing_time=self.mixing_time if self.fit_t1 else None,
            )
        )

    @fit_model.setter
    def fit_model(self, method):
        """Sets fitting model."""
        if not isinstance(method, Callable):
            error_msg = f"Fit model must be a callable object. Got: {type(method)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._fit_model = method

    @property
    def fit_function(self) -> partial:
        """Returns the fit function partially initialized."""
        # Integrity Check necessary
        return partial(
            self._fit_function,
            model=self.model,
            b_values=self.get_basis(),
            x0=self.boundaries.start_values,
            lb=self.boundaries.lower_bounds,
            ub=self.boundaries.upper_bounds,
            max_iter=self.max_iter,
            reduced=self.fit_reduced,
            mixing_time=self.mixing_time if self.fit_t1 else None,
        )

    @fit_function.setter
    def fit_function(self, method):
        """Sets fit function."""
        if not isinstance(method, (Callable, partial)):
            error_msg = f"Fit function must be a callable object. Got: {type(method)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._fit_function = method

    def get_basis(self) -> np.ndarray:
        """Calculates the basis matrix for a given set of b-values."""
        return np.squeeze(self.b_values)

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """Performs S/S0 normalization on an array"""
        img_new = np.zeros(img.shape)
        for i, j, k in zip(*np.nonzero(img[:, :, :, 0])):
            img_new[i, j, k, :] = img[i, j, k, :] / img[i, j, k, 0]
        return img_new


class IVIMSegmentedParams(IVIMParams):
    """IVIM based parameters for segmented fitting fixing one component.

    The fitting follows the classic multi exponential approach. The difference ist that one diffusion value and
    possibly the T1 value are pre fitted in the first step. The fixed diffusion value is always the first component.

    Example:
        For a tri exponential model the first component f1*exp(-b*D1) is fitted  first and the other two components are
        fitted in the second step. The fitting boundaries have to be set accordingly. It is generally recommended to
        fit the "slowest" component first.

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
        # self.fit_model = IVIM.wrapper
        self.fit_function = models.fit_curve_fixed
        # change parameters according to selected
        self.set_options(
            options.get("fixed_component", None),
            options.get("fixed_t1", False),
            options.get("reduced_b_values", None),
        )

    def init_fixed_params(self):
        self.params_fixed.model = "MonoExp"
        self.params_fixed.n_components = 1
        self.params_fixed.max_iter = self.max_iter
        self.params_fixed.n_pools = self.n_pools
        self.params_fixed.fit_reduced = self.fit_reduced

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

        # if t1 pre fitting is wanted mixing_time needs to be deployed and flag set accordingly
        if fixed_t1:
            self.params_fixed.mixing_time = self.mixing_time
            self.mixing_time = None

        if fixed_component:
            # Prepare Boundaries for the first fit
            fixed_keys = fixed_component.split("_")
            boundary_dict = dict()
            boundary_dict[fixed_keys[0]] = {}
            boundary_dict[fixed_keys[0]][fixed_keys[1]] = self.boundaries.dict[
                fixed_keys[0]
            ][fixed_keys[1]]

            # Add T1 boundaries if needed
            if fixed_t1:
                try:
                    boundary_dict["T"] = self.boundaries.dict.pop("T")
                except KeyError:
                    error_msg = "T has no defined boundaries."
                    logger.error(error_msg)
                    raise KeyError(error_msg)

            if not self.fit_reduced:
                boundary_dict["f"] = dict()
                boundary_dict["f"][fixed_keys[1]] = self.boundaries.dict["f"][
                    fixed_keys[1]
                ]

            self.params_fixed.boundaries.load(boundary_dict)

            # Prepare Boundaries for the second fit
            # Remove unused Parameter
            boundary_dict = self.boundaries.dict.copy()
            boundary_dict[fixed_keys[0]].pop(fixed_keys[1])

            # Load dict
            self.boundaries.load(boundary_dict)

        if reduced_b_values:
            self.params_fixed.b_values = reduced_b_values
        else:
            self.params_fixed.b_values = self.b_values

    def get_fixed_fit_results(self, results: list[tuple]) -> list:
        """Extract the calculated fixed values per pixel from results.

        Args:
            results (list): of tuples containing the results of the fitting process
                [0]: tuple containing pixel coordinates
                [1]: list | np.ndarray containing the fitting results

        Returns:
            [d, t1] (list): containing the fixed values for the diffusion constant and
                T1 values
        """

        d = dict()
        t_1 = dict()

        for element in results:
            d[element[0]] = element[1][1]
            if self.options["fixed_t1"]:
                t_1[element[0]] = element[1][2]

        return [d, t_1] if self.options["fixed_t1"] else [d]

    def get_pixel_args(
            self,
            img: RadImgArray | np.ndarray,
            seg: SegImgArray | np.ndarray,
            *fixed_results,
    ) -> zip:
        """Returns the pixel arguments needed for the second fitting step.

        For each pixel the coordinates (x,y,z), the corresponding pixel decay signal and the pre-fitted
        decay constant (and T1 value) are packed.

        Args:
            img (RadImgArray, np.ndarray): Nifti image
            seg (SegImgArray, np.ndarray): Segmentation image
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

    def get_pixel_args_fixed(
            self, img: RadImgArray | np.ndarray, seg: SegImgArray | np.ndarray, *args
    ) -> zip:
        """Works the same way as the IVIMParams version but can take reduced b_values
            into account.

        Args:
            img (RadImgArray, np.ndarray): Nifti image
            seg (SegImgArray, np.ndarray): Segmentation image
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
