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
import matplotlib.pyplot as plt

from radimgarray import RadImgArray, SegImgArray
from .result_dict import ResultDict
from .. import Parameters


class Results:
    """Class containing estimated diffusion values and fractions.

    Attributes:
        d (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the d values.
        f (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the f values.
        s_0 (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the S0 values.
        spectrum (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the spectrum values.
        curve (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the curve values.
        raw (ResultDict): Dict holding raw fit data.
        t_1 (ResultDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the T1 values.
        self.params (Parameters): Parameters object containing all the fitting parameters.

    Methods:
        save_results(file_path, model): Creates results dict containing pixels position,
            slice number, fitted D and f values and total number of found compartments
            and saves it as Excel sheet.

        save_spectrum(file_path): Saves spectrum of fit for every pixel as 4D Nii.

        _set_up_results_struct(self, d=None, f=None): Sets up dict containing pixel
            position, slice, d, f and number of found compartments. Used in save_results
            function.

        create_heatmap(img_dim, model, d: dict, f: dict, file_path, slice_number=0):
            Creates heatmaps for d and f in the slices segmentation and saves them as
            PNG files. If no slice_number is passed, plots the first slice.
    """

    def __init__(self, params: Parameters, **kwargs):
        """Initialize Results object."""
        self.spectrum: ResultDict = ResultDict()
        self.curve: ResultDict = ResultDict()
        self.raw: ResultDict = ResultDict()  # is this actually a thing anymore?
        self.d: ResultDict = ResultDict()
        self.f: ResultDict = ResultDict()
        self.s_0: ResultDict = ResultDict()
        self.t_1: ResultDict = ResultDict()
        self.params = params

    def set_segmentation_wise(self, identifier: dict):
        """Set segmentation info of all dicts.

        Args:
            identifier (dict): Dictionary containing pixel to segmentation value pairs.
        """
        parameters = ["spectrum", "curve", "raw", "d", "f", "s_0", "t_1"]
        for parameter in parameters:
            getattr(self, parameter).set_segmentation_wise(identifier)

    def update_results(self, results: dict):
        """Update results dict with new results."""
        for key in results.keys():
            getattr(self, key).update(results[key])

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
        for key in self.d.keys():
            row = list()
            row += self._split_or_not_to_split(
                key, split_index=split_index, is_segmentation=is_segmentation
            )

            for idx, value in enumerate([*self.d[key]]):
                rows.append(row + [f"D_{idx}", value])

            for idx, value in enumerate([*self.f[key]]):
                rows.append(row + [f"f_{idx}", value])

            rows.append(row + ["S0", self.s_0[key]])

            if self.params.TM:
                rows.append(row + ["T1", self.t_1[key]])

        column_names = self._get_column_names(
            split_index=split_index,
            is_segmentation=is_segmentation,
            additional_cols=["parameter", "value"],
        )
        df = pd.DataFrame(rows, columns=column_names)

        df.to_excel(file_path)

    def _get_column_names(
        self,
        split_index: bool = False,
        is_segmentation: bool = False,
        additional_cols: list = None,
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
            column_names += additional_cols
        return column_names

    @staticmethod
    def _split_or_not_to_split(
        key, split_index: bool = False, is_segmentation: bool = False
    ) -> row:
        """Split the key into separate columns if split_index is True."""
        row = list()
        if split_index and not is_segmentation:
            row += list(key)
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

        if not len(self.d) == 0:
            img = self.d.as_RadImgArray(img, dtype=dtype)
            img.save(file_path.parent / (file_path.stem + "_d.nii"), "nifti")
        if not len(self.f) == 0:
            img = self.f.as_RadImgArray(img, dtype=dtype)
            img.save(file_path.parent / (file_path.stem + "_f.nii"), "nifti")
        if not len(self.s_0) == 0:
            img = self.s_0.as_RadImgArray(img, dtype=dtype)
            img.save(file_path.parent / (file_path.stem + "_s0.nii"), "nifti")
        if not len(self.t_1) == 0:
            img = self.t_1.as_RadImgArray(img, dtype=dtype)
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
    ):
        """Save spectrum of fit to Excel file.

        Args:
            file_path (Path): Path to save the Excel file to.
            bins (np.ndarray, list): Bins of the spectrum.
            split_index (bool): Whether the pixel index should be split into separate
                columns.
            is_segmentation (bool): Whether the data is of a segmentation
        """
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
