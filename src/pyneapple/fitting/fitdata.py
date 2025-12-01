"""
Module combining fitting methods for pixel- and segmentation-wise  as well as segmented and ideal fitting.
Main interface for fitting methods.

Classes:
    FitData: Fitting class for (multithreaded) pixel- and segmentation-wise fitting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Type

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


from radimgarray import RadImgArray, SegImgArray

from .. import (
    IDEALParams,
    IVIMParams,
    IVIMResults,
    IVIMSegmentedParams,
    IVIMSegmentedResults,
    NNLSCVParams,
    NNLSParams,
    NNLSResults,
    Parameters,
    Results,
)
from ..utils.logger import logger
from . import fit

PARAMS_CLASSES = {
    "IVIMParams": IVIMParams,
    "IVIMSegmentedParams": IVIMSegmentedParams,
    "IDEALParams": IDEALParams,
    "NNLSParams": NNLSParams,
    "NNLSCVParams": NNLSCVParams,
}


class FitData:
    """Fitting class for (multithreaded) pixel-, segmentation-wise and segmented fitting.

    Attributes:
        img (RadImgArray): RadImgArray object with image data (4D).
        seg (SegImgArray): SegImgArray object with segmentation data (4D).
        params_json (str, Path): Path to json file with fitting parameters.

    Methods:
        fit_pixel_wise(multi_threading: bool | None = True)
            Fits every pixel inside the segmentation individually.
            Multi-threading possible to boost performance.
        fit_segmentation_wise()
            Fits mean signal of segmentation(s).
        fit_ivim_segmented(multi_threading: bool = False, debug: bool = False)
            IVIM Segmented Fitting Interface.
        fit_ideal(multi_threading: bool = False, debug: bool = False)
            IDEAL Fitting Interface.
    """

    model: Type[Parameters]
    results: Results

    def __init__(
        self,
        img: RadImgArray,  # Maybe Change signature later
        seg: SegImgArray,
        params_file: str | Path,
    ):
        """Initializes Fitting Class.

        Args:
            params_file (str, Path): Path to json/toml file with fitting parameters.
            img (RadImgArray): RadImgArray object with image data.
            seg (SegImgArray): SegImgArray object with segmentation data.
        """
        self.params_file = params_file
        self.img = img
        self.seg = seg if len(seg.shape) == 4 else np.expand_dims(seg, axis=-1)
        self._get_model()

        self.params = self.model(self.params_file)
        self._init_results()

        self.flags = dict()
        self.set_default_flags()

    @property
    def params_file(self):
        return self._params_file

    @params_file.setter
    def params_file(self, file: str | Path):
        if file is None:
            self._params_file = None
        else:
            self._params_file = Path(file)

    def _get_model(self):
        if self.params_file is not None:
            if self.params_file.suffix == ".json":
                with self.params_file.open("r") as file:
                    data = json.load(file)
            elif self.params_file.suffix == ".toml":
                with self.params_file.open("rb") as file:
                    data = tomllib.load(file)
            else:
                error_msg = "Unsupported file format! Use .json or .toml."
                logger.error(error_msg)
                raise ValueError(error_msg)
            if "Class" not in data["General"].keys():
                error_msg = "Error: Missing Class identifier!"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.model = self._get_params_class(data["General"]["Class"])

    @staticmethod
    def _get_params_class(
        class_name: str,
    ) -> Type[Parameters]:
        if class_name in PARAMS_CLASSES:
            return PARAMS_CLASSES[class_name]

        error_msg = f"Error: Invalid Class identifier '{class_name}'!"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _init_results(self):
        if isinstance(self.params, IVIMSegmentedParams):
            self.results = IVIMSegmentedResults(self.params)
        elif isinstance(self.params, IVIMParams):
            self.results = IVIMResults(self.params)
        elif isinstance(self.params, (NNLSParams, NNLSCVParams)):
            self.results = NNLSResults(self.params)
        elif isinstance(self.params, IDEALParams):
            self.results = IVIMResults(self.params)
        else:
            error_msg = "Error: Invalid Parameter Class Identifier!"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def set_default_flags(self):
        """Sets default flags for fitting class."""
        self.flags["did_fit"] = False

    def _print_fit_info(self, fit_type: str):
        fit_info = f"Fitting {fit_type}:\n"
        fit_info += f"  - Image: {self.img}\n"
        fit_info += f"  - Segmentation: {self.seg}\n"
        fit_info += f"  - Fit Type: {self.params.fit_type}\n"

    def fit_pixel_wise(self, fit_type: str | None = None, **kwargs):
        """Fits every pixel inside the segmentation individually.

        Args:
            fit_type (str): (optional) Type of fitting to be used (single, multi, gpu).
            kwargs (dict): Additional keyword arguments to pass to the fit function.
        """
        self._print_fit_info("Pixelwise")

        results = fit.fit_pixel_wise(self.img, self.seg, self.params, fit_type)
        if results is not None:
            self.results.eval_results(results)

    def fit_segmentation_wise(self):
        """Fits mean signal of segmentation(s), computed of all pixels signals."""
        if self.img is None or self.seg is None:
            error_msg = "No valid data for fitting selected!"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._print_fit_info("Segmentationwise")
        results = fit.fit_segmentation_wise(self.img, self.seg, self.params)

        seg_indices = dict()
        for seg_number in self.seg.seg_values:
            indices = self.seg.squeeze(axis=3).get_seg_indices(seg_number)
            seg_indices.update(
                {
                    key: value
                    for (key, value) in zip(indices, [seg_number * 1] * len(indices))
                }
            )

        self.results.set_segmentation_wise(seg_indices)
        self.results.eval_results(results)

    def fit_ivim_segmented(
        self, fit_type: str | None = None, debug: bool = False, **kwargs
    ):
        """IVIM Segmented Fitting Interface.
        Args:
            fit_type (str): (optional) Type of fitting to be used (single, multi, gpu).
            debug (bool): If True, debug output is printed.
            kwargs (dict): Additional keyword arguments to pass to the fit function.
        """
        if not isinstance(self.params, IVIMSegmentedParams):
            error_msg = "Invalid Parameter Class for IVIM Segmented Fitting!"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._print_fit_info("Segmented")

        fixed_component, results = fit.fit_ivim_segmented(
            self.img, self.seg, self.params, fit_type, debug, **kwargs
        )
        # Evaluate Results
        self.results.eval_results(results, fixed_component=fixed_component)

    def fit_ideal(self, fit_type: str | None = None, debug: bool = False):
        """IDEAL Fitting Interface.
        Args:
            fit_type (str): (optional) Type of fitting to be used (single, multi, gpu).
            debug (bool): (optional) If True, debug output is printed.
        """

        if not isinstance(self.params, IDEALParams):
            error_msg = "Invalid Parameter Class for IDEAL Fitting!"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._print_fit_info("IDEAL")

        fit_results = fit.fit_ideal(self.img, self.seg, self.params, fit_type, debug)
        self.results.eval_results(fit_results)
