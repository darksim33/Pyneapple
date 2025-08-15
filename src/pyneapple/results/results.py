"""Base module for handling and saving fit results.

This module contains the Results class, which is used to store and save the results of
a fit. The class contains methods to save the results to an Excel file, a NifTi file,
or as a heatmap plot.

Classes:
    Results: Class containing estimated diffusion values and fractions. This is the base
        class for the corresponding Result classes for each Fitting Procedure.
"""

from __future__ import annotations

from abc import abstractmethod

from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.logger import logger
from radimgarray import RadImgArray
from .result_dict import ResultDict
from .. import Parameters


class BaseResults:
    """Class containing estimated diffusion values and fractions.

    Attributes:
        D (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the d values.
        f (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the f values.
        S0 (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the S0 values.
        spectrum (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the spectrum values.
        curve (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the curve values.
        raw (ResultDict): Dict holding raw fit data.
        t1 (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the T1 values.
        self.params (Parameters): Parameters object containing all the fitting parameters.

    Methods:
        save_to_excel(file_path: Path, split_index: bool, is_segmentation: bool): 
            Creates results dict containing pixels position, slice number, fitted D and 
            f values and total number of found compartments and saves it as Excel sheet.

        save_to_nii(file_path: Path, img: RadImgArray, dtype=int, separate_files=False, 
            **kwargs):
            Saves all fitted parameters to NIfTi files. If separate_files is True, each
            parameter (D1, D2,...) is saved in a separate file, otherwise all parameters 
            are saved in one file (4th dimension holding D1, D2,...). 

        save_spectrum_to_nii(file_path: Path | str, img: RadImgArray):
            Saves spectrum of fit for every pixel as 4D Nii.

        save_spectrum_to_excel(file_path: Path | str, bins: np.ndarray | list,
                                split_index: bool = False, is_segmentation: bool = False, **kwargs):
            Saves spectrum of fit to Excel file.      

        save_fit_curve_to_excel(file_path: Path | str, b_values: np.ndarray,
                                 split_index: bool = False, is_segmentation: bool = False):
            Saves curve of fit to Excel file.
    """

    def __init__(self, params: Parameters, **kwargs):
        """Initialize Results object."""
        self.spectrum: ResultDict = ResultDict()
        self.curve: ResultDict = ResultDict()
        self.raw: ResultDict = ResultDict()  # is this actually a thing anymore?
        self.D: ResultDict = ResultDict()
        self.f: ResultDict = ResultDict()
        self.S0: ResultDict = ResultDict()
        self.t1: ResultDict = ResultDict()
        self.params = params

    def set_segmentation_wise(self, identifier: dict):
        """Set segmentation info of all dicts.

        Args:
            identifier (dict): Dictionary containing pixel to segmentation value pairs.
        """
        parameters = ["spectrum", "curve", "raw", "D", "f", "S0", "t1"]
        for parameter in parameters:
            getattr(self, parameter).set_segmentation_wise(identifier)

    def update_results(self, results: dict):
        """Update results dict with new results."""
        for key in results.keys():
            getattr(self, key).update(results[key])

    @abstractmethod
    def eval_results(self, results: list, **kwargs):
        """Evaluate the results."""
        pass

    def save_to_excel(
        self, file_path: Path, split_index: bool = False, is_segmentation: bool = False
    ):
        """Save the results to an Excel file.

        Args:
            file_path (Path): Path to save the Excel file to.
            split_index (bool): Whether the pixel index should be split into separate
                columns.
            is_segmentation (bool): Whether the data is of a segmentation or not.
        """

        # Creating a list of lists where each list is a row in the Excel file.
        rows = list()
        for key in self.D.keys():
            row = list()
            row += self._split_or_not_to_split(
                key, split_index=split_index, is_segmentation=is_segmentation
            )
            rows = self._get_row_data(row, rows, key)

        column_names = self._get_column_names(
            split_index=split_index,
            is_segmentation=is_segmentation,
            additional_cols=["parameter", "value"],
        )
        df = pd.DataFrame(rows, columns=column_names)

        df.to_excel(file_path)

    def _get_row_data(self, row: list, rows: list, key) -> list:
        for idx, value in enumerate([*self.D[key]]):
            rows.append(row + [f"D_{idx}", value])

        for idx, value in enumerate([*self.f[key]]):
            rows.append(row + [f"f_{idx}", value])

        rows.append(row + ["S0", self.S0[key]])
        return rows

    def _get_column_names(
        self,
        split_index: bool = False,
        is_segmentation: bool = False,
        additional_cols: np.ndarray | list | None = None,
    ) -> list:
        """Get the column names for the Excel file.

        Args:
            split_index (bool): Whether the pixel index should be split into separate
                columns.
            is_segmentation (bool): Whether the data is of a segmentation or not.
            additional_cols (list): Additional columns to add to the column names.
        """
        if not is_segmentation:
            if split_index:
                column_names = ["x", "y", "z"]
            else:
                column_names = ["pixel"]
        else:
            column_names = ["seg_number"]
        if additional_cols:
            if isinstance(additional_cols, np.ndarray):
                if len(additional_cols.shape) == 1:
                    column_names += additional_cols.tolist()
                elif len(additional_cols.shape) == 2 and additional_cols.shape[1] == 1:
                    column_names += np.squeeze(additional_cols).tolist()
                else:  # no cover
                    error_msg = "Additional columns should be a 1D array or a 2D array with one column."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                column_names += additional_cols
        return column_names

    @staticmethod
    def _split_or_not_to_split(
        key, split_index: bool = False, is_segmentation: bool = False
    ) -> list:
        """Split the key into separate columns if split_index is True."""
        row = list()
        if split_index and not is_segmentation:
            row += list(key)
        elif not split_index and is_segmentation:
            row += [key]
        else:
            row += [[*key]]
        return row

    def save_to_nii(
        self,
        file_path: Path,
        img: RadImgArray,
        dtype: object | None = int,
        separate_files: bool = False,
        **kwargs,
    ):
        """Save all fitted parameters to NIfTi files.

        Args:
            file_path (Path): Path to save the NIfTi files to including the basic name.
            img (RadImgArray): RadImgArray object containing the image data.
            dtype (object): Data type of the NIfTi files.
            separate_files (bool): Whether to save each parameter in a separate file.
        """
        if not separate_files:
            self._save_non_separated_nii(file_path, img, dtype, **kwargs)
        else:
            self._save_separate_nii(file_path, img, dtype, **kwargs)

    @abstractmethod
    def _save_separate_nii(
        self, file_path: Path, img: RadImgArray, dtype: object | None = int, **kwargs
    ):
        """Save each parameter in a separate NIfTi file.

        Args:
            file_path (Path): Path to save the NIfTi files to including the basic name.
            img (RadImgArray): RadImgArray image the fit was performed on.
            dtype (object): Data type of the NIfTi files.
            **kwargs: Additional options for saving the data.
        """
        pass

    def _save_non_separated_nii(
        self, file_path: Path, img: RadImgArray, dtype: object | None = int, **kwargs
    ):
        """Each NIfTi contains all diffusion values or fractions (or S0 or T1) for each
        pixel.

        Args:
            file_path (Path): Path to save the NIfTi files to including the basic name.
            img (RadImgArray): RadImgArray object containing the image data.
            dtype (object): Data type of the NIfTi files.
        """

        if not len(self.D) == 0:
            img = self.D.as_RadImgArray(img, dtype=dtype)
            img.save(file_path.parent / (file_path.stem + "_d.nii"), "nifti")
        if not len(self.f) == 0:
            img = self.f.as_RadImgArray(img, dtype=dtype)
            img.save(file_path.parent / (file_path.stem + "_f.nii"), "nifti")
        if not len(self.S0) == 0:
            img = self.S0.as_RadImgArray(img, dtype=dtype)
            img.save(file_path.parent / (file_path.stem + "_s0.nii"), "nifti")
        if not len(self.t1) == 0:
            img = self.t1.as_RadImgArray(img, dtype=dtype)
            img.save(file_path.parent / (file_path.stem + "_t1.nii"), "nifti")

    def save_spectrum_to_nii(self, file_path: Path | str, img: RadImgArray):
        """Saves spectrum of fit for every pixel as 4D Nii.

        Args:
            file_path (Path): Path to save the NIfTi files to including the basic name.
            img (RadImgArray): RadImgArray object containing the image data.
        """
        spec = self.spectrum.as_RadImgArray(img)
        spec.save(file_path, save_as="nii")

    def save_spectrum_to_excel(
        self,
        file_path: Path | str,
        bins: np.ndarray | list,
        split_index: bool = False,
        is_segmentation: bool = False,
        **kwargs,
    ):
        """Save spectrum of fit to Excel file.

        Args:
            file_path (Path): Path to save the Excel file to.
            bins (np.ndarray, list): Bins of the spectrum.
            split_index (bool, optional): Whether the pixel index should be split into
                separate columns.
            is_segmentation (bool, optional): Whether the data is of a segmentation
            **kwargs: Additional options for saving the data.
        """
        split_index = kwargs.get("split_index", False)
        is_segmentation = kwargs.get("is_segmentation", False)

        if isinstance(bins, np.ndarray):
            bins = bins.tolist()

        rows = list()
        for key in self.spectrum.keys():
            row = list()
            row += self._split_or_not_to_split(
                key, split_index=split_index, is_segmentation=is_segmentation
            )
            row += np.squeeze(self.spectrum[key]).tolist()
            rows.append(row)

        column_names = self._get_column_names(
            split_index=split_index,
            is_segmentation=is_segmentation,
            additional_cols=bins,
        )

        df = pd.DataFrame(rows, columns=column_names)
        df.to_excel(file_path)

    def save_fit_curve_to_excel(
        self,
        file_path: Path | str,
        b_values: np.ndarray,
        split_index: bool = False,
        is_segmentation: bool = False,
    ):
        """Save curve of fit to Excel file.

        Args:
            file_path (Path): Path to save the Excel file to.
            b_values (np.ndarray): B values of the curve.
            split_index (bool): Whether the pixel index should be split into separate.
            is_segmentation (bool): Whether the data is of a segmentation.
        """
        if isinstance(b_values, np.ndarray):
            b_values = b_values.tolist()

        rows = list()
        for key in self.curve.keys():
            row = list()
            row += self._split_or_not_to_split(
                key, split_index=split_index, is_segmentation=is_segmentation
            )
            row += np.squeeze(self.curve[key]).tolist()
            rows.append(row)

        column_names = self._get_column_names(
            split_index=split_index,
            is_segmentation=is_segmentation,
            additional_cols=b_values,
        )

        df = pd.DataFrame(rows, columns=column_names)
        df.to_excel(file_path)
