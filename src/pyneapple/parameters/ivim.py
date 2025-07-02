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

from ..models import MonoExpFitModel
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
        if self.fit_t1 and not self.mixing_time:
            error_msg = "T1 mapping is set but no mixing time is defined."
            logger.error(error_msg)
            raise ValueError(error_msg)

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
        if not "exp" in model.lower():
            error_msg = f"Only exponential models are supported. Got: {model}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            if "mono" in model.lower():
                self.n_components = 1
                self.fit_model = models.MonoExpFitModel(model)
            elif "bi" in model.lower():
                self.n_components = 2
                self.fit_model = models.BiExpFitModel(model)
            elif "tri" in model.lower():
                self.n_components = 3
                self.fit_model = models.TriExpFitModel(model)
            else:
                error_msg = f"Only mono-, bi- and tri-exponential models are supported atm. Got: {model}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        self._model = model.upper()

    @property
    def fit_model(self):
        """Return fit model with set parameters."""
        try:
            self._fit_model.reduced = self.fit_reduced
        except AttributeError:
            error_msg = "Fit model does not have a 'reduced' attribute."
            logger.warning(error_msg)
        try:
            self._fit_model.fit_t1 = self.fit_t1
            self._fit_model.mixing_time = self.mixing_time
        except AttributeError:
            error_msg = "Fit model does not have 'fit_t1' or 'mixing_time' attributes."
            logger.warning(error_msg)
        return self._fit_model.model


    @fit_model.setter
    def fit_model(self, model):
        """Sets fitting model."""
        self._fit_model = model

    @property
    def fit_function(self) -> partial:
        """Returns the fit function partially initialized."""
        # Integrity Check necessary

        return partial(
            self._fit_model.fit,
            model=self._fit_model.model,
            b_values=self.get_basis(),
            x0=self.boundaries.start_values,
            lb=self.boundaries.lower_bounds,
            ub=self.boundaries.upper_bounds,
            max_iter=self.max_iter,
            reduced=self.fit_reduced,
            mixing_time=self.mixing_time if self.fit_t1 else None,
        )

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
    ):
        """
        Multi-exponential Parameter class used for the segmented IVIM fitting.

        Args:
            params_json (str | Path): Json containing fitting parameters

        Additional Parameters (compared to IVIMParams):
            fixed_component (str):  In the shape of "D_slow" or "D_1" with "_" as
                separator for dict
            fixed_t1 (bool, optional): Set T1 for pre fitting, defaults to False
                If T1 is used in the first fitting step, its passed as a konstant in the
                second fitting step.
            reduced_b_values (list, optional): of b_values used for first fitting (second
                fitting is always performed with all)

        """
        self.fixed_component = None
        self.fixed_t1 = False
        self.reduced_b_values = None

        super().__init__(params_json)
        # Set mono / ADC default params set as starting point
        self.params_fixed = IVIMParams()
        self.init_fixed_params()
        if self.fixed_component:
            self.set_up()

    @property
    def fixed_component(self):
        return self._fixed_component

    @fixed_component.setter
    def fixed_component(self, value: str):
        """Sets the fixed component for segmented fitting."""
        if value is None:
            self._fixed_component = None
        elif isinstance(value, str):
            if "_" in value:
                self._fixed_component = value
            else:
                error_msg = "Fixed component must be in the form 'D_slow' or 'D_1'."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = "Fixed component must be a string."
            logger.error(error_msg)
            raise TypeError(error_msg)

    @property
    def fixed_t1(self):
        return self._fixed_t1

    @fixed_t1.setter
    def fixed_t1(self, value: bool):
        """Sets the flag for T1 mapping in segmented fitting."""
        if isinstance(value, bool):
            self._fixed_t1 = value
        else:
            error_msg = "Fixed T1 must be a boolean value."
            logger.error(error_msg)
            raise TypeError(error_msg)

    @property
    def reduced_b_values(self):
        return self._reduced_b_values

    @reduced_b_values.setter
    def reduced_b_values(self, values: list | np.ndarray):
        if isinstance(values, list):
            values = np.array(values)
        elif values is None:
            self._reduced_b_values = np.array([])
        if isinstance(values, np.ndarray):
            self._reduced_b_values = np.expand_dims(values.squeeze(), axis=1)

    def init_fixed_params(self):
        self.params_fixed.model = "MonoExp"
        self.params_fixed.n_components = 1
        self.params_fixed.max_iter = self.max_iter
        self.params_fixed.n_pools = self.n_pools
        self.params_fixed.fit_reduced = self.fit_reduced

    def set_up(self):
        """Set options for segmented fitting based on the parameters."""

        # Check if fixed component is valid and add to temp dictionary
        fixed_keys = self.fixed_component.split("_")
        # prepare boundaries for the first fit
        _dict = self.boundaries.dict.get(fixed_keys[0], {})
        _value = _dict.get(fixed_keys[1], None)
        if _value is not None:
            _dict = {fixed_keys[0]: {}}
            _dict[fixed_keys[0]][fixed_keys[1]] = self.boundaries.dict[
                fixed_keys[0]
            ][fixed_keys[1]]
        else:
            error_msg = (f"Fixed component {self.fixed_component} is not valid. "
                         f"No corresponding boundaries found in the parameter set.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.fixed_t1:
            if not self.fit_t1:
                error_msg = "T1 mapping is set but not enabled in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif not self.mixing_time:
                error_msg = "Mixing time is set but not passed in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.params_fixed.mixing_time = self.mixing_time
            self.params_fixed.fit_t1 = self.fit_t1
            self.fit_t1 = False
            _dict["T"] = self.boundaries.dict.pop("T", {})
            if not _dict["T"]:
                error_msg = "T1 has no defined boundaries."
                logger.error(error_msg)
                raise KeyError(error_msg)
            # TODO add t1 to fixed params???

        self.params_fixed.boundaries.load(_dict)

        # Prepare boundaries for the second fit
        _dict = self.boundaries.dict.copy()
        _dict[fixed_keys[0]].pop(fixed_keys[1])
        self.boundaries.load(_dict)

        self.params_fixed.b_values = self.reduced_b_values if self.reduced_b_values.any() else self.b_values


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

        if self.fixed_t1:
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
        if not self.reduced_b_values.any():
            pixel_args = super().get_pixel_args(img, seg, args)
        else:
            # get b_value positions
            indexes = np.where(
                np.isin(
                    self.b_values,
                    self.reduced_b_values,
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
