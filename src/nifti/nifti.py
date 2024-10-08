from __future__ import annotations

import imantics
import numpy as np
import nibabel as nib
import pandas as pd

from typing import Type
from pathlib import Path
from copy import deepcopy
from PIL import Image


class Nii:
    """
    Class based on NiBabels NifTi-class with additional functionality

    ...

    Attributes
    ----------
    path : Path
        File path to nifti file
    array : np.ndarray
        Image array
    affine : np.ndarray
        Image Rotation Matrix
    header : nib.Nifti1Header()
        NifTi header information
    # mask : bool
    #     Mask or anatomical image

    Methods
    ----------
    load(path=None)
        Load NifTi image
    save()
        Save to NifTi image
    reset()
        Reload NifTi image
    copy()
        copy Nii
    show(slice)
        Show image of selected Slice as PNG
    from_array(array)
        Create NifTi object from np.ndarray
    to_rgba_array(slice, alpha)
        Return RGBA ndarray
    to_rgba_image(slice, alpha)
        Return RGBA PIL.Image
    QPixmap(slice, scaling)
        Return QPixmap from Slice

    """

    def __init__(self, path: str | Path | None = None, **kwargs) -> None:
        self._original_scaling = None
        self.path = path
        # self.__set_path(path)
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        # self.mask: bool = False
        self.__load()
        for key in kwargs:
            if key == "do_zero_padding" and kwargs["do_zero_padding"]:
                self.zero_padding()

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, value: str | Path | None):
        if isinstance(value, str):
            value = Path(value)
        # elif value is None:
        #     # value = Path.cwd() / ".temp.nii" # for handling of the save process
        #     pass
        self._path = value

    def load(self, path: Path | str = None):
        """Load NifTi file."""
        self.__load(path)

    def __load(self, path: Path | str | None = None) -> None:
        """Private Loader"""
        if self.path is None:
            if path:
                self.path = path
            else:
                return None

        if self.path.is_file():
            nifti = nib.load(self.path)
            self.array = np.array(nifti.get_fdata())
            while len(self.array.shape) < 4:
                self.array = np.expand_dims(self.array, axis=-1)
            self.affine = nifti.affine
            self.header = nifti.header
        else:
            print("File not found!")
            return None

    def reset(self):
        """Resets Nii by loading the file again"""
        self.__load()

    def clear(self):
        """Removes all data from obj"""

        self.path = None
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()

    def setter(self, img: Nii):
        """
        Transfer data from one Nii object to another.

        This is useful if you don't want to change the PyObject but can't work on the object itself.
        """
        self.path = img.path
        self.array = img.array
        self.affine = img.affine
        self.header = img.header

    def zero_padding(self):
        if self.array.shape[0] < self.array.shape[1]:
            new_array = np.pad(
                self.array,
                (
                    (
                        int((self.array.shape[1] - self.array.shape[0]) / 2),
                        int((self.array.shape[1] - self.array.shape[0]) / 2),
                    ),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
            )
            self.array = new_array
        elif self.array.shape[1] < self.array.shape[0]:
            new_array = np.pad(
                self.array,
                (
                    (0, 0),
                    (
                        int((self.array.shape[0] - self.array.shape[1]) / 2),
                        int((self.array.shape[0] - self.array.shape[1]) / 2),
                    ),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
            )
            self.array = new_array
        self.header.set_data_shape(self.array.shape)

    def save(self, name: str | Path, dtype: Type = int, do_zip: bool = True):
        """
        Save Nii to File

        Attributes:
            name (str|Path): Name of the output file to save the data to.
            dtype (object): Sets the output data type of the NifTi (int and float supported)
            do_zip (bool): "Will force the zipping of the file
        """

        save_path = self.path.parent / name if self.path is not None else name
        # Save as Int/float
        array = np.array(self.array.astype(dtype).copy())
        header = self.header
        if dtype == int:
            header.set_data_dtype("i4")
        elif dtype == float:
            header.set_data_dtype("f4")
        new_Nii = nib.Nifti1Image(
            array,
            self.affine,
            header,
        )
        # https://note.nkmk.me/en/python-numpy-dtype-astype/
        # https://brainder.org/2012/09/23/the-nifti-file-format/
        nib.save(new_Nii, save_path)

    def copy(self):
        """Make Copy of Nii class obj"""

        return deepcopy(self)

    def from_array(
        self,
        array: np.ndarray,
        header: nib.Nifti1Header | None = None,
        path: str | Path | None = None,
    ):
        """Create Nii image with a given or default header"""

        self.array = array
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header() if not header else header
        self.path = path
        return self

    def to_rgba_array(self, slice_number: int = 0, alpha: int = 1) -> np.ndarray:
        """Return RGBA array"""

        # Return RGBA array of Nii
        # rot Image
        array = (
            np.rot90(self.array[:, :, slice_number, 0])
            if slice_number is not None
            else self.array[:, :, 0, 0]
        )
        if not np.nanmax(array) == np.nanmin(array):
            array_norm = (array - np.nanmin(array)) / (
                np.nanmax(array) - np.nanmin(array)
            )
        else:
            print("Divided by zero.Setting to 1.")
            array_norm = array / np.nanmax(array)
        alpha_map = np.ones(array_norm.shape)
        array_rgba = np.dstack((array_norm, array_norm, array_norm, alpha_map))

        return array_rgba

    # Might be unnecessary by now. Only works with plotting.overlay_image
    def to_rgba_image(self, slice_number: int = 0, alpha: int = 1) -> Image:
        # Return RGBA PIL Image of Nii slice
        array_rgba = self.to_rgba_array(slice_number, alpha) * 255
        img_rgba = Image.fromarray(
            array_rgba.round().astype(np.int8).copy(),  # Needed for Image
            "RGBA",
        )

        return img_rgba

    def show(self, slice: int | None = None, tag: str = "array"):
        """Show image as matplotlib figure or PNG"""

        if tag == "png" or tag == "image":
            img_rgb = self.to_rgba_image(slice)
            img_rgb.show()
        elif tag == "array":
            array_rgb = self.to_rgba_array(slice)
            fig, ax = plt.subplots()
            ax.imshow(array_rgb[:, :, :])
            plt.set_cmap("gray")
            plt.show()

    def scale_image(self, scaling: str | None | int = None):
        """
        Scale image.

        Parameters:
            scaling (str): Scaling option. Options are "S/S0" or a scalar
        """
        if scaling == "S/S0":
            array = self.array.copy()
            s0 = array[:, :, :, 0]
            self._original_scaling = s0
            new_array = np.divide(array, s0[:, :, :, np.newaxis])
            self.array = new_array
        elif isinstance(scaling, int) and not isinstance(scaling, bool):
            self.array = self.array * scaling


class NiiFit(Nii):
    def __init__(
        self,
        path: str | Path | None = None,
        n_components: int | np.ndarray | None = 1,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.n_components = n_components
        self.d_weight = 10000
        self.f_weight = 100
        self.s0_weight = 1
        self.scaling = kwargs.get("scaling", None)

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scale: np.ndarray | list | None = None):
        if scale is None:
            scaling = np.zeros(2 * self.n_components + 1)
            scaling[: self.n_components] = self.d_weight
            scaling[self.n_components : -1] = self.f_weight
            scaling[-1] = self.s0_weight
        elif isinstance(scale, np.ndarray):
            scaling = scale
        elif isinstance(scale, (list, tuple)):
            scaling = np.array(scale)
        else:
            scaling = None
        self._scaling = scaling

    def save(
        self,
        file_name: str | Path,
        dtype: object = int,
        save_type: str = "single",
        parameter_names: list | None = None,
        do_zip: bool = True,
    ) -> None:
        """
        Save array and save as int (float is optional but not recommended).

        Attributes:

        name: str
            File name of the saved file. (without parent Path)
        dtype: object
            Defines data type for nifti.
        save_type: str
            Defines what kind of Save is chosen.
            "single": all Data ist saved to a single file with the fourth dimension representing variables.
        parameter_names: list
            List of Parameter Names
        do_zip: bool
            Whether files should be zipped.


        Information:

        Nifti Header Codes:
            https://note.nkmk.me/en/python-numpy-dtype-astype/
            https://brainder.org/2012/09/23/the-nifti-file-format/
        """

        def _create_new_name(
            file_path: Path, variable_name: str, dozip: bool
        ) -> Path | None:
            if file_path.suffix == ".nii":
                if dozip:
                    out_path = file_path.parent / (
                        file_path.stem + variable_name + ".nii.gz"
                    )
                else:
                    out_path = file_path.parent / (
                        file_path.stem + variable_name + ".nii"
                    )
            elif file_path.suffix == ".gz":
                if not Path(file_path.stem).suffix == "":
                    if dozip:
                        out_path = file_path.parent / (
                            Path(file_path.stem).stem + ".nii.gz"
                        )
                    else:
                        out_path = file_path.parent / (
                            Path(file_path.stem).stem + ".nii"
                        )
            else:
                out_path = None
                ValueError("File type not supported!")
            return out_path

        save_path = self.path.parent / file_name if self.path is not None else file_name
        if "gz" not in save_path.suffix and do_zip:
            save_path = save_path.with_suffix(save_path.suffix + ".gz")
        if save_type == "single":
            array = self.scale_image_all().astype(dtype)
            header = self.header
            if dtype == int:
                header.set_data_dtype("i4")
            elif dtype == float:
                header.set_data_dtype("f4")
            new_nii = nib.Nifti1Image(
                array,
                self.affine,
                header,
            )
            nib.save(new_nii, save_path)
        elif save_type == "separate":
            for comp in range(len(parameter_names)):
                array = self.scale_image_single_variable(
                    self.array[:, :, :, comp], self.scaling[comp]
                ).astype(dtype)
                header = self.header
                if dtype == int:
                    header.set_data_dtype("i4")
                elif dtype == float:
                    header.set_data_dtype("f4")
                new_nii = nib.Nifti1Image(
                    array,
                    self.affine,
                    header,
                )
                if parameter_names:
                    var_name = parameter_names[comp]
                else:
                    var_name = comp

                save_path_new = _create_new_name(file_name, var_name, do_zip)
                print(f"Saving to: {save_path_new}")
                nib.save(new_nii, save_path_new)

    def scale_image_all(self) -> np.ndarray | None:
        """Scale all variables to clinical dimensions"""
        array = self.array.copy()
        if isinstance(self.n_components, int):
            scaling = np.zeros(2 * self.n_components + 1)
            scaling[: self.n_components] = self.d_weight
            scaling[self.n_components : -1] = self.f_weight
            scaling[-1] = self.s0_weight
            array_scaled = array * scaling
        elif isinstance(self.n_components, np.ndarray):
            array_scaled = None
        else:
            array_scaled = None
        return array_scaled

    @staticmethod
    def scale_image_single_variable(
        array: np.ndarray, scale: int | float | np.ndarray
    ) -> np.ndarray | None:
        """Scale a single variable to clinical dimensions"""
        if isinstance(scale, (int, float)):
            array_scaled = array * scale
        elif isinstance(scale, np.ndarray):
            array_scaled = None
        else:
            array_scaled = None
        return array_scaled
