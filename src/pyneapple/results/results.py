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
from . import ResultDict
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
        parameters = ["spectrum", "curve", "raw", "d", "f", "S0", "T1"]
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
            if split_index:
                row += list(*key)
            else:
                row += list(key)
            row += [*self.d[key]]
            row += [*self.f[key]]
            row += [*self.s_0[key]]
            if self.params.TM:
                row += [*self.t_1[key]]
            rows.append(row)

        column_names = self._get_column_names(
            split_index=split_index, is_segmentation=is_segmentation
        )
        df = pd.DataFrame(rows, columns=column_names)

        df.save_to_excel(file_path)

    @abstractmethod
    def _get_column_names(
        self, split_index: bool = False, is_segmentation: bool = False
    ) -> list:
        """Get the column names for the Excel file."""
        if split_index:
            column_names = ["x", "y", "z"]
        else:
            column_names = ["pixel"]
        return column_names

    @abstractmethod
    def save_to_nii(
        self,
        file_path: Path,
        img: RadImgArray,
        dtype: object | None = int,
        seperate_files: bool = False,
        **kwargs,
    ):
        """Save all fitted parameters to NIfTi files.

        Args:
            file_path (Path): Path to save the NIfTi files to including the basic name.
            img (RadImgArray): RadImgArray object containing the image data.
            dtype (object): Data type of the NIfTi files.
            seperate_files (bool): Whether to save each parameter in a separate file.
        """
        if not seperate_files:
            self._save_nonseperated_nii(file_path, img, dtype, **kwargs)
        else:
            self._save_seperate_nii(file_path, img, dtype, **kwargs)

    def _save_nonseperated_nii(
        self, file_path: Path, img: RadImgArray, dtype: object | None = int, **kwargs
    ):
        """Each NIfTi contains all diffusion values or fractions (or S0 or T1) for each pixel."""

        if not len(self.d) == 0:
            array = self.d.as_RadImgArray(img, dtype=dtype)
            array.save(file_path.parent / (file_path.stem + "_d.nii"), "nifti")
        if not len(self.f) == 0:
            array = self.f.as_RadImgArray(img, dtype=dtype)
            array.save(file_path.parent / (file_path.stem + "_f.nii"), "nifti")
        if not len(self.s_0) == 0:
            array = self.s_0.as_RadImgArray(img, dtype=dtype)
            array.save(file_path.parent / (file_path.stem + "_s0.nii"), "nifti")
        if not len(self.t_1) == 0:
            array = self.t_1.as_RadImgArray(img, dtype=dtype)
            array.save(file_path.parent / (file_path.stem + "_t1.nii"), "nifti")

    @abstractmethod
    def _save_seperate_nii(
        self, file_path: Path, img: RadImgArray, dtype: object | None = int, **kwargs
    ):
        pass

    # def save_results_to_excel(
    #     self,
    #     file_path: Path | str,
    #     d: dict = None,
    #     f: dict = None,
    #     split_index=False,
    #     is_segmentation=False,
    # ):
    #     """Saves the results of a model fit to an Excel file.

    #     Args:
    #         file_path (str): The path where the Excel file will be saved.
    #         d (dict): Optional argument. Sets diffusion coefficients to save if
    #             different from fit results.
    #         f (dict): Optional argument. Sets volume fractions to save if different from
    #             fit results.
    #         split_index (bool): Whether the pixel index should be split into separate
    #             columns.
    #         is_segmentation (bool): Whether the data is of a segmentation or not.
    #     """
    #     # Set d and f as current fit results if not passed
    #     if not (d or f):
    #         d = self.d
    #         f = self.f

    #     df = pd.DataFrame(
    #         self._set_up_results_dict(
    #             d, f, split_index=split_index, is_segmentation=is_segmentation
    #         )
    #     ).T

    #     if split_index and not is_segmentation:
    #         # Restructure key index into columns and save results
    #         # df.reset_index(names=["pixel_x", "pixel_y", "slice"], inplace=True)
    #         df = df.rename(
    #             columns={
    #                 "element_key_0": "pixel_x",
    #                 "element_key_1": "pixel_y",
    #                 "element_key_2": "slice",
    #             }
    #         )
    #     elif is_segmentation:
    #         df = df.rename(columns={"element_key": "seg_number"})
    #         # df.reset_index(names=["seg_number"], inplace=True)
    #     else:
    #         df = df.rename(columns={"element_key": "pixel_index"})
    #         # df.reset_index(names=["pixel_index"], inplace=True)
    #     df = self._sort_column_names(df)
    #     df.to_excel(file_path)

    # def _set_up_results_dict(
    #     self, d: dict, f: dict, split_index: bool = False, is_segmentation: bool = False
    # ) -> dict:
    #     """Sets up dict containing pixel position, slice, d, f and number of found
    #     compartments.

    #     Args:
    #         d (dict): Optional argument. Sets diffusion coefficients to save if
    #             different from fit results.
    #         f (dict): Optional argument. Sets volume fractions to save if different
    #             from fit results.
    #         split_index (bool): Whether the pixel index should be split into separate
    #             columns.
    #         is_segmentation (bool): Whether the data is of a segmentation or not.
    #     Returns:
    #         result_dict (dict): Dictionary containing pixel position, slice, d, f and
    #             number of found compartments.
    #     """
    #     # Set d and f as current fit results if not passed
    #     if not (d or f):
    #         d = self.d
    #         f = self.f

    #     result_dict = {}
    #     element_idx = 0
    #     pixel_idx = 0
    #     for key, d_values in d.items():
    #         n_comp = len(d_values)
    #         pixel_idx += 1
    #         for comp, d_comp in enumerate(d_values):
    #             if is_segmentation:
    #                 result_dict[element_idx] = {"element_key": key}
    #             elif split_index:
    #                 result_dict[element_idx] = {
    #                     "element_key_0": key[0],
    #                     "element_key_1": key[1],
    #                     "element_key_2": key[2],
    #                 }
    #             else:
    #                 result_dict[element_idx] = {
    #                     "element_key": str(key)
    #                     .replace("(", "")
    #                     .replace(")", "")
    #                     .replace(" ", "")
    #                 }

    #             result_dict[element_idx].update(
    #                 {
    #                     "element": pixel_idx,
    #                     "D": d_comp,
    #                     "f": f[key][comp],
    #                     "compartment": comp + 1,
    #                     "n_compartments": n_comp,
    #                 }
    #             )
    #             element_idx += 1
    #     return result_dict

    # @staticmethod
    # def _sort_column_names(df: pd.DataFrame) -> pd.DataFrame:
    #     """Sort column names.

    #     Args:
    #         df (pd.DataFrame): Containing data for saving with wrong column order.

    #     Returns:
    #         df (pd.DataFrame): Containing data for saving with correct column order.
    #     """
    #     main_labels = ["element", "D", "f", "compartment", "n_compartments"]
    #     current_labels = df.columns.tolist()
    #     additional_labels = [
    #         element for element in current_labels if element not in main_labels
    #     ]
    #     new_labels = (
    #         additional_labels + main_labels
    #     )  # assuming the missing labels are the ones specific to indexing style
    #     df = df.reindex(columns=new_labels)
    #     return df

    # def save_fitted_parameters_to_nii(
    #     self,
    #     file_path: str | Path,
    #     shape: tuple,
    #     parameter_names: list | dict | None = None,
    #     dtype: object | None = int,
    # ):
    #     """Saves the results of a IVIM fit to an NifTi file.

    #     Args:
    #         file_path (str): The path where the NifTi file will be saved.
    #         shape (tuple): Contains 3D matrix size of original Image.
    #         dtype (object): Handles datatype to save the NifTi in. int and float are
    #             supported.
    #         parameter_names (list | dict): Containing the variables as keys and names as
    #             items (list of str).
    #     """
    #     file_path = Path(file_path) if isinstance(file_path, str) else file_path

    #     # determine number of parameters
    #     n_components = len(self.d[next(iter(self.d))])

    #     if len(shape) >= 4:
    #         shape = shape[:3]

    #     array = np.zeros((shape[0], shape[1], shape[2], n_components * 2 + 1))
    #     # Create Parameter Maps Matrix
    #     for key in self.d:
    #         array[key[0], key[1], key[2], 0:n_components] = self.d[key]
    #         array[key[0], key[1], key[2], n_components:-1] = self.f[key]
    #         array[key[0], key[1], key[2], -1] = self.s_0[key]

    #     # TODO: needs rework
    #     # out_nii = NiiFit(n_components=n_components).from_array(array)

    #     names = list()

    #     # Check for Parameter Names
    #     if parameter_names is not None:
    #         if isinstance(parameter_names, dict):
    #             names = list()
    #             for key in parameter_names:
    #                 names = names + [key + item for item in parameter_names[key]]
    #         elif isinstance(parameter_names, list):
    #             names = parameter_names
    #         else:
    #             ValueError("Parameter names must be a dict or a list")

    #     # out_nii.save(
    #     #     file_path, dtype=dtype, save_type="separate", parameter_names=names
    #     # )
    #     print("All files have been saved.")

    def save_spectrum_to_nii(self, file_path: Path | str, img: RadImgArray):
        """Saves spectrum of fit for every pixel as 4D Nii."""
        spec = self.spectrum.as_RadImgArray(img)
        spec.save(file_path, save_as="nii")

    def save_spectrum_to_excel(self, bins: np.ndarray, file_path: Path | str):
        """Save spectrum of fit to Excel file."""
        _dict = {"index": bins.tolist()}
        _dict.update(self.spectrum)
        df = pd.DataFrame(_dict).T
        df.columns = df.iloc[0]
        df = df[1:]
        df.to_excel(file_path)

    def save_fit_curve_to_excel(self, b_values: np.ndarray, file_path: Path | str):
        """Save curve of fit to Excel file."""
        _dict = {"index": b_values.squeeze().tolist()}
        curve = self.curve
        for key in curve:
            curve[key] = curve[key].squeeze()
        _dict.update(curve)
        df = pd.DataFrame(_dict).T
        df.columns = df.iloc[0]
        df = df[1:]
        df.to_excel(file_path)

    @staticmethod
    def create_heatmap(fit_data, file_path: Path | str, slice_numbers: int | list):
        """Creates heatmap plots for d and f results of pixels inside the segmentation,
        saved as PNG.

        N heatmaps are created dependent on the number of compartments up to the
        tri-exponential model. Needs d and f to be of same length throughout whole
        struct. Used in particular for AUC results.

        Args:
            fit_data (FitData): Object holding model, img and seg information.
            file_path (str): The path where the Excel file will be saved.
            slices_contain_seg (int, list): Number of slice(s) heatmap should be created
                of.
        """
        if isinstance(slice_numbers, int):
            slice_numbers = [slice_numbers]

        # Apply AUC (for smoothed results with >3 components)
        (d, f) = fit_data.params.apply_AUC_to_results(fit_data.results)
        img_dim = fit_data.img.array.shape[0:3]

        # Check first pixels result for underlying number of compartments
        n_comps = len(d[list(d)[0]])

        model = fit_data.model_name

        for slice_number, slice_contains_seg in enumerate(slice_numbers):
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
