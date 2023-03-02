import numpy as np
import nibabel as nib
from PyQt6.QtGui import QPixmap, QImage
from scipy import ndimage
from typing import Tuple


class appData:
    def __init__(self):
        self.plt_boundries: np.array = np.array([0.0001, 0.2])
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
    def __init__(self) -> None:
        nifti_img.array = np.array
        nifti_img.affine = np.array
        nifti_img.header = np.array
        nifti_img.size = np.array

    def load(self, path: str) -> None:
        nii = nib.load(path)
        nifti_img.array = nii.get_fdata()
        nifti_img.affine = nii.affine
        nifti_img.header = nii.header
        nifti_img.size = nii.shape

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
