"""Module for handling and saving fit results.

This module contains the Results class, which is used to store and save the results of
a fit. The class contains methods to save the results to an Excel file, a NifTi file,
or as a heatmap plot.

Classes:
    CustomDict: Custom dictionary for storing fitting results and returning them
        according to fit style.
    Results: Class containing estimated diffusion values and fractions.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from nifti import NiiFit, Nii

if TYPE_CHECKING:
    from . import FitData


class CustomDict(dict):
    """Custom dictionary for storing fitting results and returning them according to
    fit style.

    Basic dictionary enhanced with fit type utility to store results of pixel or
    segmentation wise fitting. Some methods are overwritten to handle either of the
    two fit types.

    Attributes:
        fit_type (str): Type of fit process. Either "Pixel" or "Segmentation".
        identifier (dict): Dictionary containing pixel to segmentation value pairs.

    Methods:
        set_segmentation_wise(self, identifier: dict) Update the dictionary for
            segmented fitting
        as_array(self, shape: dict) -> np.ndarray Return array containing the dict
            content
    """

    def __init__(self, fit_type: str | None = None, identifier: dict | None = None):
        """Initialize CustomDict object.

        Args:
            fit_type (str): Type of fit process. Either "Pixel" or "Segmentation".
            identifier (dict): Dictionary containing pixel to segmentation value pairs.
        """
        super().__init__()
        self.type = fit_type
        self.identifier = identifier
        if fit_type == "Segmentation" and identifier is None:
            raise ValueError("Identifier is required if fit_type is 'Segmentation'")

    def __getitem__(self, key):
        """Return value of key in dictionary.

        The __getitem__ method is overwritten to handle either pixel or segmentation
        based keys.

        Args:
            key: Key to look up in dictionary.
        Returns:
            value: Value of key in dictionary.
        """
        value = None
        key = self.validate_key(key)
        if isinstance(key, tuple):
            # If the key is a tuple containing the pixel coordinates:
            try:
                if self.type == "Segmentation":
                    # in case of Segmentation wise fitting the identifier
                    # dict is needed to look up pixel segmentation number
                    value = super().__getitem__(self.identifier[key])
                else:
                    value = super().__getitem__(key)
            except KeyError:
                raise KeyError(f"Key '{key}' not found in dictionary.")
        elif isinstance(key, int):
            # If the key is an int for the segmentation:
            try:
                if self.type == "Segmentation":
                    value = super().__getitem__(key)
            except KeyError:
                raise KeyError(f"Key '{key}' not found in dictionary.")
        return value

    def __setitem__(self, key, value):
        """Set value of key in dictionary."""
        key = self.validate_key(key)
        super().__setitem__(key, value)

    def get(self, key, default=None):
        """Return value of key in dictionary."""
        try:
            return self.__getitem__(key)
        except KeyError:
            if default is not None:
                return default
            else:
                KeyError(f"Key '{key}' not found in dictionary.")

    @staticmethod
    def validate_key(key):
        """Validate key type.
        Check weather the given key is supported by the CustomDict."""
        if isinstance(key, tuple):
            pass
        elif isinstance(key, int):
            pass
        elif isinstance(key, str):
            TypeError("String assignment and calling is not supported.")
        elif isinstance(key, float):
            TypeError("Float assignment and calling is not supported.")
        try:
            if np.issubdtype(key, np.integer):
                key = int(key)
        except TypeError:
            pass
        return key

    def set_segmentation_wise(self, identifier: dict | None = None):
        """
        Update segmentation info of dict.

        Args:
            identifier (dict): Dictionary containing pixel to segmentation value pairs.
        """
        if isinstance(identifier, dict):
            self.identifier = identifier  # .copy()
            self.type = "Segmentation"
        elif identifier is None or False:
            self.identifier = {}
            self.type = "Pixel"

    def as_array(self, shape: tuple | list) -> np.ndarray:
        """Returns a numpy array of the dict fit data.

        Args:
            shape (tuple): Shape of final fit data.

        Returns:
            array (np.ndarray): Numpy array of the dict fit data.
        """
        if isinstance(shape, tuple):
            shape = list(shape)

        if len(shape) < 4:
            ValueError("Shape must be at least 4 dimensions.")
        elif shape[3] == 1:
            shape[3] = list(self.values())[0].shape[
                0
            ]  # read shape of first array in dict to determine shape
            pass
        array = np.zeros(shape)

        if self.type == "Segmentation":
            for key, seg_number in self.identifier.items():
                array[key] = self[seg_number]
        else:
            for key, value in self.items():
                array[key] = value
        return array


class Results:
    """Class containing estimated diffusion values and fractions.

    Attributes:
        d (CustomDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the d values.
        f (CustomDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the f values.
        S0 (CustomDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the S0 values.
        spectrum (CustomDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the spectrum values.
        curve (CustomDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the curve values.
        raw (CustomDict): Dict holding raw fit data.
        T1 (CustomDict): Dict of tuples containing pixel coordinates as keys and a
            np.ndarray holding all the T1 values.

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

    def __init__(self):
        """Initialize Results object."""
        self.spectrum: CustomDict = CustomDict()
        self.curve: CustomDict = CustomDict()
        self.raw: CustomDict = CustomDict()
        self.d: CustomDict = CustomDict()
        self.f: CustomDict = CustomDict()
        self.S0: CustomDict = CustomDict()
        self.T1: CustomDict = CustomDict()

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

    def save_results_to_excel(
        self,
        file_path: Path | str,
        d: dict = None,
        f: dict = None,
        split_index=False,
        is_segmentation=False,
    ):
        """Saves the results of a model fit to an Excel file.

        Args:
            file_path (str): The path where the Excel file will be saved.
            d (dict): Optional argument. Sets diffusion coefficients to save if
                different from fit results.
            f (dict): Optional argument. Sets volume fractions to save if different from
                fit results.
            split_index (bool): Whether the pixel index should be split into separate
                columns.
            is_segmentation (bool): Whether the data is of a segmentation or not.
        """
        # Set d and f as current fit results if not passed
        if not (d or f):
            d = self.d
            f = self.f

        df = pd.DataFrame(
            self._set_up_results_dict(
                d, f, split_index=split_index, is_segmentation=is_segmentation
            )
        ).T

        if split_index and not is_segmentation:
            # Restructure key index into columns and save results
            # df.reset_index(names=["pixel_x", "pixel_y", "slice"], inplace=True)
            df = df.rename(
                columns={
                    "element_key_0": "pixel_x",
                    "element_key_1": "pixel_y",
                    "element_key_2": "slice",
                }
            )
        elif is_segmentation:
            df = df.rename(columns={"element_key": "seg_number"})
            # df.reset_index(names=["seg_number"], inplace=True)
        else:
            df = df.rename(columns={"element_key": "pixel_index"})
            # df.reset_index(names=["pixel_index"], inplace=True)
        df = self._sort_column_names(df)
        df.to_excel(file_path)

    def _set_up_results_dict(
        self, d: dict, f: dict, split_index: bool = False, is_segmentation: bool = False
    ) -> dict:
        """Sets up dict containing pixel position, slice, d, f and number of found
        compartments.

        Args:
            d (dict): Optional argument. Sets diffusion coefficients to save if
                different from fit results.
            f (dict): Optional argument. Sets volume fractions to save if different
                from fit results.
            split_index (bool): Whether the pixel index should be split into separate
                columns.
            is_segmentation (bool): Whether the data is of a segmentation or not.
        Returns:
            result_dict (dict): Dictionary containing pixel position, slice, d, f and
                number of found compartments.
        """
        # Set d and f as current fit results if not passed
        if not (d or f):
            d = self.d
            f = self.f

        result_dict = {}
        element_idx = 0
        pixel_idx = 0
        for key, d_values in d.items():
            n_comp = len(d_values)
            pixel_idx += 1
            for comp, d_comp in enumerate(d_values):
                if is_segmentation:
                    result_dict[element_idx] = {"element_key": key}
                elif split_index:
                    result_dict[element_idx] = {
                        "element_key_0": key[0],
                        "element_key_1": key[1],
                        "element_key_2": key[2],
                    }
                else:
                    result_dict[element_idx] = {
                        "element_key": str(key)
                        .replace("(", "")
                        .replace(")", "")
                        .replace(" ", "")
                    }

                result_dict[element_idx].update(
                    {
                        "element": pixel_idx,
                        "D": d_comp,
                        "f": f[key][comp],
                        "compartment": comp + 1,
                        "n_compartments": n_comp,
                    }
                )
                element_idx += 1
        return result_dict

    @staticmethod
    def _sort_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Sort column names.

        Args:
            df (pd.DataFrame): Containing data for saving with wrong column order.

        Returns:
            df (pd.DataFrame): Containing data for saving with correct column order.
        """
        main_labels = ["element", "D", "f", "compartment", "n_compartments"]
        current_labels = df.columns.tolist()
        additional_labels = [
            element for element in current_labels if element not in main_labels
        ]
        new_labels = (
            additional_labels + main_labels
        )  # assuming the missing labels are the ones specific to indexing style
        df = df.reindex(columns=new_labels)
        return df

    def save_fitted_parameters_to_nii(
        self,
        file_path: str | Path,
        shape: tuple,
        parameter_names: list | dict | None = None,
        dtype: object | None = int,
    ):
        """Saves the results of a IVIM fit to an NifTi file.

        Args:
            file_path (str): The path where the NifTi file will be saved.
            shape (tuple): Contains 3D matrix size of original Image.
            dtype (object): Handles datatype to save the NifTi in. int and float are
                supported.
            parameter_names (list | dict): Containing the variables as keys and names as
                items (list of str).
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

        names = list()

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

        out_nii.save(
            file_path, dtype=dtype, save_type="separate", parameter_names=names
        )
        print("All files have been saved.")

    def save_spectrum_to_nii(self, file_path: Path | str, shape: tuple):
        """Saves spectrum of fit for every pixel as 4D Nii."""
        spec = Nii().from_array(self.spectrum.as_array(shape))
        spec.save(file_path)

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
    def create_heatmap(
        fit_data: FitData, file_path: Path | str, slice_numbers: int | list
    ):
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
