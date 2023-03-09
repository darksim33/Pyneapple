import numpy as np
import nibabel as nib
from pathlib import Path

# import pandas as pd
import math, csv
from PyQt6.QtGui import QPixmap, QImage
from scipy import ndimage
from typing import Tuple


class appData:
    def __init__(self):
        self.plt_boundries: np.ndarray = np.ndarray([0.0001, 0.2])
        self.plt_nslice: int = nslice()
        self.plt_scaling: int = 2
        self.imgDyn: nifti_img = nifti_img()


class nslice:
    def __init__(self):
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


class nifti_img:
    def __init__(self, path: str | Path | None = None) -> None:
        self.set_path(path)
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.array
        self.header = np.array
        self.size = np.array
        self.__load()

    def reset(self):
        self.__load()

    def save(self, name: str | Path):
        save_path = self.path.parent / name

    def set_path(self, path: str | Path):
        self.path = Path(path) if path is not None else None

    def load(self, path: Path | str):
        self.set_path(path)
        self.__load()

    def __load(self) -> None:
        if self.path is None:
            return None
        nii = nib.load(self.path)
        self.array = nii.get_fdata()
        while len(self.array.shape) < 4:
            self.array = np.expand_dims(self.array, axis=-1)
        self.affine = nii.affine
        self.header = nii.header
        self.size = self.array.shape

    def nii2QPixmap(self, slice: int, scaling: int) -> QPixmap:
        img = np.rot90(self.array[:, :, slice, 0])
        img_norm = (img - img.min()) / (img.max() - img.min())
        img_zoomed = ndimage.zoom(img_norm, (scaling, scaling), order=0, mode="nearest")
        img_rgb = (
            (np.dstack((img_zoomed, img_zoomed, img_zoomed)) * 255)
            .round()
            .astype("int8")
            .copy()
        )
        qimg = QImage(
            img_rgb,
            img_rgb.shape[1],
            img_rgb.shape[0],
            img_rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        qpixmap = QPixmap.fromImage(qimg)
        return qpixmap


def applyMask2Image(img: nifti_img, mask: nifti_img):
    if img.size[0:2] == mask.size[0:2]:
        # img_masked = nifti_img()
        img_masked = img
        mask.array[mask.array == 0] = math.nan
        for idx in range(img.size[3]):
            img_masked.array[:, :, :, idx] = np.multiply(
                img.array[:, :, :, idx], mask.array[:, :, :, 0]
            )
        return img_masked


def Signal2CSV(img: nifti_img, path: str | None = None):
    data = dict()
    for idx in range(img.size[0]):
        for idy in range(img.size[1]):
            for idz in range(img.size[2]):
                if not math.isnan(img.array[idx, idy, idz, 0]):
                    data["".join((str(x) + " ") for x in [idx, idy, idz])] = img.array[
                        idx, idy, idz, :
                    ]
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in data.items():
            writer.writerow([key, value])


def lbl2npcoord(ypos: int, ysize: int, scaling: int):
    # y Axis is inverted for label coordinates
    new_pos = ysize * scaling - ypos
    return new_pos


def np2lblcoord(
    xpos: int, ypos: int, xsize: int, ysize: int, scaling: int
) -> Tuple[int, int]:
    new_x_pos = int(xpos / scaling)

    # y Axis is inverted for label coordinates
    new_y_pos = ysize - int(ypos / scaling)
    return new_x_pos, new_y_pos


def display_img(nii, axis, slice, scaling):
    axis.setPixmap(nii.nii2QPixmap(slice, scaling))
