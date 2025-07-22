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
import numpy as np
from copy import deepcopy

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

    def __init__(self, file: str | Path | None = None):
        self.boundaries = IVIMBoundaries()
        self.n_components = 0
        self.fit_reduced = False
        self.fit_S0 = False
        self.fit_t1 = False
        self.mixing_time = None
        super().__init__(file)
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
        self._set_model(model)

    def _set_model(self, model: str):
        """Sets the model for the IVIM fitting."""
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
    def fit_function(self) -> partial:
        """Returns the fit function partially initialized."""
        # Integrity Check necessary

        return partial(
            self.fit_model.fit,
            model=self.fit_model.model,
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
        params_1 (IVIMParams): Fixed parameters for segmented fitting.
        options (dict): Options for segmented fitting.
    Methods:
        set_options: Setting necessary options for segmented IVIM fitting.
        get_fixed_fit_results: Extract the calculated fixed values per pixel from
            results.
        get_pixel_args: Returns the pixel arguments needed for the second fitting step.
        get_pixel_args_fixed: Works the same way as the IVIMParams version but can take
            reduced b_values into account.
        eval_fitting_results: Assigns fit results to the diffusion parameters D & f.
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
        self.params_1 = IVIMParams()
        self.params_2 = IVIMParams()

        super().__init__(params_json)
        # Set mono / ADC default params set as starting point
        self._init_params()
        if self.fixed_component:
            self.set_up()

    @property
    def model(self):
        """Get or set the model for the segmented fitting."""
        return self._model

    @model.setter
    def model(self, model: str):
        self._set_model(model)
        if model.lower() == "triexp" and self.params_2.model != "BiExp":
            self._init_params()

    @property
    def fixed_component(self):
        return self._fixed_component

    @fixed_component.setter
    def fixed_component(self, value: str):
        """Sets the fixed component for segmented fitting."""
        if value is None:
            self._fixed_component = None
        elif isinstance(value, str):
            if "_" in value and len(value.split("_")) == 2:
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
        """Sets the flag for T1 mapping in first fitting step and usage of fitted T1
        values as parameters for second fit."""
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

    def _init_params(self):
        """Initialize the parameter subsets for the segmented fitting."""
        self.params_1.model = "MonoExp"
        self.params_1.n_components = 1
        self.params_1.max_iter = self.max_iter
        self.params_1.n_pools = self.n_pools
        self.params_1.fit_reduced = self.fit_reduced

        if self.model.lower() == "triexp":
            self.params_2.model = "BiExp"
            self.params_2.n_components = 2
        else:  # default to mono exponential
            self.params_2.model = "MonoExp"
            self.params_2.n_components = 1
        self.params_2.max_iter = self.max_iter
        self.params_2.n_pools = self.n_pools
        self.params_2.fit_reduced = self.fit_reduced

    def set_up(self):
        """
        Set options for segmented fitting based on the parameters.
        Is used after parameters are set manually to prepare first and second fit.
        """

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

        if not self.fit_reduced:
            _dict.update({"f": {fixed_keys[1]: self.boundaries.dict["f"][fixed_keys[1]]}})

        if self.fixed_t1:
            if not self.fit_t1:
                error_msg = "T1 mapping is set but not enabled in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif not self.mixing_time:
                error_msg = "Mixing time is set but not passed in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.params_1.mixing_time = self.mixing_time
            self.params_1.fit_t1 = True
            self.params_1._fit_model.fit_t1 = True
            self.params_1._fit_model.mixing_time = self.mixing_time
            # self.fit_t1 = False
            _dict["T"] = self.boundaries.dict.get("T", {})
            if not _dict["T"]:
                error_msg = "T1 has no defined boundaries."
                logger.error(error_msg)
                raise KeyError(error_msg)
            # TODO add t1 to fixed params???

        self.params_1.boundaries.load(_dict)

        # Prepare boundaries for the second fit
        _dict = deepcopy(self.boundaries.dict)
        _dict[fixed_keys[0]].pop(fixed_keys[1])
        if self.fixed_t1:
            _dict.pop("T")
            self.params_2.fit_t1 = False
            self.params_2._fit_model.fit_t1 = False
        elif self.fit_t1:
            self.params_2.fit_t1 = True
            self.params_2._fit_model.fit_t1 = True
            self.params_2._fit_model.mixing_time = self.mixing_time
            if not self.mixing_time:
                error_msg = "Mixing time is set but not passed in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.params_2.mixing_time = self.mixing_time

        self.params_2.boundaries.load(_dict)

        # Set reduced b_values if available
        self.params_1.b_values = self.reduced_b_values if self.reduced_b_values.any() else self.b_values
        self.params_2.b_values = self.b_values

    def get_fixed_fit_results(self, results: list[tuple]) -> list:
        """Extract the calculated fixed values per pixel from results.

        Args:
            results (list): of tuples containing the results of the fitting process
                [0]: tuple containing pixel coordinates
                [1]: list | np.ndarray containing the fitting results

        Returns:
            [D, t1] (list): containing the fixed values for the diffusion constant and
                T1 values
        """

        d = dict()
        t_1 = dict()

        for element in results:
            d[element[0]] = element[1][1]
            if self.fixed_t1:
                t_1[element[0]] = element[1][2]

        return [d, t_1] if self.fixed_t1 else [d]

    def get_pixel_args_fit1(
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

    def get_pixel_args_fit2(
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

    def _prepare_data_for_saving(self) -> dict:
        """Prepare data for saving to json."""
        data = super()._prepare_data_for_saving()
        data.pop("params_1", None)
        data.pop("params_2", None)
        return data
