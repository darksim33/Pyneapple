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
from .boundaries import IVIMBoundaryDict
from .. import models
from radimgarray import RadImgArray, SegImgArray


class IVIMParams(BaseParams):
    """
    Multi-exponential Parameter class used for the IVIM model.

    Attributes:
        boundaries (IVIMBoundaryDict): Boundaries for IVIM model.
        mixing_time (bool): Flag for T1 mapping.

    Methods:
        set_boundaries(): Sets lower and upper fitting boundaries and starting values
            for IVIM.
    """

    def __init__(self, file: str | Path | None = None):
        self.fit_model = models.BaseExpFitModel()
        self.boundaries = IVIMBoundaryDict()
        super().__init__(file)
        if self.fit_model.fit_t1 and not self.fit_model.mixing_time:
            error_msg = "T1 mapping is set but no mixing time is defined."
            logger.error(error_msg)
            raise ValueError(error_msg)

    @property
    def model(self):
        model = self.fit_model.name
        if hasattr(self.fit_model, "fit_reduced") and self.fit_model.fit_reduced:
            model += "_red"
        elif hasattr(self.fit_model, "fit_S0") and self.fit_model.fit_S0:
            model += "_S0"
        if self.fit_model.fit_t1:
            model += "_T1"
        return model.upper()

    def _set_model(self, model: str):
        """Sets the model for the IVIM fitting."""
        if not model and isinstance(model, str):
            return
        if "exp" not in model.lower():
            error_msg = f"Only exponential models are supported. Got: {model}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            if "base" in model.lower():
                self.fit_model = models.BaseExpFitModel(model)
            elif "mono" in model.lower():
                self.fit_model = models.MonoExpFitModel(model)
            elif "bi" in model.lower():
                self.fit_model = models.BiExpFitModel(model)
            elif "tri" in model.lower():
                self.fit_model = models.TriExpFitModel(model)
            else:
                error_msg = f"Only mono-, bi- and tri-exponential models are supported atm. Got: {model}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _set_model_parameters(self, model_params: dict):
        """Sets model parameters from a dictionary.
        Make adjustments for fit_model creation.

        Args:
            model_params (dict): Dictionary containing model parameters.
        """
        if "model" in model_params or "name" in model_params:
            self._set_model(model_params["model"])
            super()._set_model_parameters(model_params)
        elif "n_components" in model_params:
            model_params.pop("n_components")
        else:
            error_msg = "Model parameters must contain 'model' or 'name' key."
            logger.error(error_msg)
            raise KeyError(error_msg)

    def _prepare_data_for_saving(self) -> dict:
        _dict = super()._prepare_data_for_saving()
        if "n_components" in _dict["Model"].keys():
            _dict["Model"].pop("n_components")
        return _dict

    @property
    def fit_function(self) -> partial:
        """Returns the fit function partially initialized."""
        # Integrity Check necessary

        return partial(
            self.fit_model.fit,
            b_values=self.get_basis(),
            x0=self.boundaries.start_values,
            lb=self.boundaries.lower_bounds,
            ub=self.boundaries.upper_bounds,
            max_iter=self.max_iter,
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
            fit_reduced b_values into account.
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
        self.fixed_component = ""
        self.fixed_t1 = False
        self.reduced_b_values = np.array([])
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
    def fixed_component(self) -> str:
        return self._fixed_component

    @fixed_component.setter
    def fixed_component(self, value: str):
        """Sets the fixed component for segmented fitting."""
        if value is None or value == "":
            self._fixed_component = ""
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
        self.params_1._set_model("MonoExp")
        self.params_1.max_iter = self.max_iter
        self.params_1.n_pools = self.n_pools
        self.params_1.fit_model.fit_reduced = self.fit_model.fit_reduced

        if self.fit_model.name:
            self.params_2._set_model(self.fit_model.name)
        self.params_2.max_iter = self.max_iter
        self.params_2.n_pools = self.n_pools
        self.params_2.fit_model.fit_reduced = self.fit_model.fit_reduced
        self.params_2.fit_model.fit_S0 = (
            self.fit_model.fit_S0 if hasattr(self.fit_model, "fit_S0") else False
        )

    def set_up(self):
        """
        Set options for segmented fitting based on the parameters.
        Is used after parameters are set manually to prepare first and second fit.
        """

        # Check if fixed component is valid and add to temp dictionary
        fixed_keys = self.fixed_component.split("_")
        if fixed_keys[1].isnumeric():
            self.params_2.fit_model.fix_d = int(fixed_keys[1])
        else:
            error_msg = (
                f"Fixed component {self.fixed_component} is not valid. "
                f"Only numeric indices are allowed."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # prepare boundaries for the first fit
        _dict = self.boundaries.get(fixed_keys[0], {})
        _value = _dict.get(fixed_keys[1], None)
        if _value is not None:
            _dict = {fixed_keys[0]: {}}
            _dict[fixed_keys[0]][fixed_keys[1]] = self.boundaries[fixed_keys[0]][
                fixed_keys[1]
            ]
        else:
            error_msg = (
                f"Fixed component {self.fixed_component} is not valid. "
                f"No corresponding boundaries found in the parameter set."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not self.fit_model.fit_reduced:
            _dict.update(self._get_s0_boundaries())

        if self.fixed_t1:
            if not self.fit_model.fit_t1:
                error_msg = "T1 mapping is set but not enabled in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif not self.fit_model.mixing_time:
                error_msg = "Mixing time is set but not passed in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.params_1.fit_model.fit_t1 = True
            self.params_1.fit_model.mixing_time = self.fit_model.mixing_time
            # self.fit_t1 = False
            _dict["T"] = self.boundaries.get("T", {})
            if not _dict["T"]:
                error_msg = "T1 has no defined boundaries."
                logger.error(error_msg)
                raise KeyError(error_msg)
            # TODO add t1 to fixed params???

        self.params_1.boundaries = IVIMBoundaryDict(_dict)

        # Prepare boundaries for the second fit
        _dict = deepcopy(self.boundaries)
        _dict[fixed_keys[0]].pop(fixed_keys[1])
        if self.fixed_t1:
            _dict.pop("T")
            self.params_2.fit_model.fit_t1 = False
        elif self.fit_model.fit_t1:
            self.params_2.fit_model.fit_t1 = True
            self.params_2.fit_model.mixing_time = self.fit_model.mixing_time
            if not self.fit_model.mixing_time:
                error_msg = "Mixing time is set but not passed in the parameters."
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.params_2.fit_model.mixing_time = self.fit_model.mixing_time

        self.params_2.boundaries = IVIMBoundaryDict(_dict)

        # Set fit_reduced b_values if available
        self.params_1.b_values = (
            self.reduced_b_values if self.reduced_b_values.any() else self.b_values
        )
        self.params_2.b_values = self.b_values

    def _get_s0_boundaries(self) -> dict:
        """Returns the S0 boundaries for the first fitting process."""
        if hasattr(self.fit_model, "fit_S0") and self.fit_model.fit_S0:
            result = self.boundaries.get("S", {}).get("0", {})
        else:
            fractions = self.boundaries.get("f", {})
            result = np.array([])
            for key in fractions:
                array = np.array(fractions[key])
                if result.size == 0:
                    result = array
                else:
                    result += array
            result = result.tolist()
        return {"S": {"0": result}}

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
            d[element[0]] = element[1][0]
            if self.fixed_t1:
                t_1[element[0]] = element[1][2]

        return [d, t_1] if self.fixed_t1 else [d]

    def get_pixel_args_fit1(
        self, img: RadImgArray | np.ndarray, seg: SegImgArray | np.ndarray, *args
    ) -> zip:
        """Works the same way as the IVIMParams version but can take fit_reduced b_values
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
                (
                    (int(i), int(j), int(k))
                    for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
                ),
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

        indexes = [
            (int(i), int(j), int(k))
            for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
        ]
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
        data["General"].pop("params_1", None)
        data["General"].pop("params_2", None)
        return data
