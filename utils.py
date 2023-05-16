import numpy as np
import nibabel as nib

# import pandas as pd
import warnings
from pathlib import Path
from PIL import Image, ImageQt  # , ImageFilter, ImageOps
import matplotlib.pyplot as plt

from copy import deepcopy
from PyQt6.QtGui import QPixmap
from PyQt6 import QtCore


class Nii:
    """
    Class based on NiBabels NifTi-class with additonal functionility

    ...

    Attributes
    ----------
    path : Path
        Path to nifti file
    array : np.ndarray
        Image array
    affine : np.ndarray
        Image Rotation Matrix
    header : nib.Nifti1Header()
        NifTi header information
    mask : bool
        Mask or anatomical image

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
    fromArray(array)
        Create NifTi object from np.ndarray
    to_rgba_array(slice, alpha)
        Return RGBA ndarray
    to_rgba_image(slice, alpha)
        Return RGBA PIL.Image
    QPixmap(slice, scaling)
        Return QPixmap from Slice

    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.__set_path(path)
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        self.mask: bool = False
        self.__load()

    def load(self, path: Path | str):
        self.__set_path(path)
        self.__load()

    def __load(self) -> None:
        if self.path is None:
            return None
        nifti = nib.load(self.path)
        self.array = np.array(nifti.get_fdata())
        while len(self.array.shape) < 4:
            self.array = np.expand_dims(self.array, axis=-1)
        self.affine = nifti.affine
        self.header = nifti.header

    def __set_path(self, path: str | Path):
        self.path = Path(path) if path is not None else None

    def reset(self):
        self.__load()

    def save(self, name: str | Path, dtype: object = int):
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
        return deepcopy(self)

    def fromArray(self, array: np.ndarray, ismask: bool = False):
        """Create Nii image with a default header"""
        self.set_path = None
        self.array = array
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        self.mask = True if ismask else False
        return self

    def to_rgba_array(self, slice: int = 0, alpha: int = 1) -> np.ndarray:
        # Return RGBA array of Nii
        # rot Image
        array = (
            np.rot90(self.array[:, :, slice, 0])
            if slice is not None
            else self.array[:, :, 0, 0]
        )
        # Add check for empty mask
        array_norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        # if nifti is mask -> Zeros get zero alpha
        alpha_map = array * alpha if self.mask else np.ones(array.shape)
        array_rgba = np.dstack((array_norm, array_norm, array_norm, alpha_map))

        return array_rgba

    def to_rgba_image(self, slice: int = 0, alpha: int = 1) -> Image:
        # Return RGBA PIL Image of Nii slice
        array_rgba = self.to_rgba_array(slice, alpha) * 255
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
            plt.show()

    def QPixmap(self, slice: int = 0, scaling: int = 1) -> QPixmap:
        if self.path:
            img = self.to_rgba_image(slice).copy()
            img = img.resize(
                [img.size[0] * scaling, img.size[1] * scaling],
                Image.NEAREST,
            )
            qPixmap = QPixmap.fromImage(ImageQt.ImageQt(img))
            return qPixmap
        else:
            return None


class Nii_seg(Nii):
    # Nii segmentation image: kann be a mask or a ROI based nifti image
    def __init__(self, path: str | Path | None = None):
        super().__init__(path)
        self.mask = True
        self._nSegs = np.unique(self.array).max() if self.path is not None else None

    @property
    def nSegs(self):
        """Number of Segmentations"""
        if self.path:
            self._nSegs = np.unique(self.array).max()
        return self._nSegs

    def get_segIndizes(self, index):
        idxs = np.array(np.where(self.array == index))
        return idxs

    def evaluate_seg(self):
        print("Evaluating Segmentation")


class NSlice:
    def __init__(self, value: int = None):
        if not value:
            self._value = value
            self._number = value + 1
        else:
            self._value = None
            self._number = None

    @property
    def number(self):
        return self._number

    @property
    def value(self):
        return self._value

    @number.setter
    def number(self, value):
        self._number = value
        self._value = value - 1

    @value.setter
    def value(self, value):
        self._number = value + 1
        self._value = value


class Processing(object):
    def merge_nii_images(img1: Nii, img2: Nii) -> Nii:
        array1 = img1.array.copy()
        array2 = img2.array.copy()
        if img2.mask:
            if np.array_equal(array1.shape[0:2], array2.shape[0:2]):
                # compare inplane size of Arrays
                array_merged = np.ones(array1.shape)
                for idx in range(img1.array.shape[3]):
                    array_merged[:, :, :, idx] = np.multiply(
                        array1[:, :, :, idx], array2[:, :, :, 0]
                    )
                img_merged = img1.copy()
                img_merged.array = array_merged
                return img_merged
        else:
            warnings.warn("Warning: Secondary Image is not a mask!")


class IndexTracker:
    def __init__(self, ax, X):
        self.index = 0
        self.X = X
        self.ax = ax
        self.im = ax.imshow(self.X[:, :, self.index], cmap="gray")
        self.update()

    def on_scroll(self, event):
        # print(event.button, event.step)
        increment = 1 if event.button == "up" else -1
        max_index = self.X.shape[-1] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.index])
        self.ax.set_title(f"Use scroll wheel to navigate\nindex {self.index}")
        self.im.axes.figure.canvas.draw()
