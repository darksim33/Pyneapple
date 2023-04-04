import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps, ImageQt, ImageFilter
from copy import deepcopy
import math
from PyQt6.QtGui import QPixmap, QImage
from PyQt6 import QtCore
from scipy import ndimage
from typing import Tuple
from matplotlib import pyplot as plt


class nifti_img:
    def __init__(self, path: str | Path | None = None) -> None:
        self.set_path(path)
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.array
        self.header = np.array
        self.size = np.array
        self.mask: bool = False
        self.__load()

    def reset(self):
        self.__load()

    def save(self, name: str | Path):
        save_path = self.path.parent / name
        # Save as Int
        new_nii = nib.Nifti1Image(
            # self.get_fdata().astype("int32"), self.affine, self.header  # wtf
            self.array.astype("int32"),
            self.affine,
            self.header,
        )
        new_nii.set_data_dtype(
            8
        )  # https://brainder.org/2012/09/23/the-nifti-file-format/
        nib.save(new_nii, save_path)

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
        self.size = np.array(self.array.shape)

    def copy(self):
        return deepcopy(self)

    def show(self, slice: int | None = None):
        img_rgb = self.rgb(slice)
        img_rgb.show()

    def rgb(self, slice: int | None = None):
        # Used anymore?
        array = (
            self.array[:, :, slice, 0] if slice is not None else self.array[:, :, 0, 0]
        )
        array_norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        img_rgb = Image.fromarray(
            (np.dstack((array_norm, array_norm, array_norm)) * 255)
            .round()
            .astype("int8")
            .copy(),
            "RGB",
        )
        return img_rgb

    def rgba(self, slice: int = 0, alpha: int = 1) -> Image:
        # Return RGBA PIL Image of nii slice
        # rot Image
        array = (
            np.rot90(self.array[:, :, slice, 0])
            if slice is not None
            else self.array[:, :, 0, 0]
        )
        array_norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        # if nifti is mask -> Zeros get zero alpha
        alpha_map = array * alpha if self.mask else np.ones(array.shape)
        img_rgba = Image.fromarray(
            (np.dstack((array_norm, array_norm, array_norm, alpha_map)) * 255)
            .round()
            .astype("int8")
            .copy(),
            "RGBA",
        )

        return img_rgba

    def QPixmap(self, slice: int = 0, scaling: int = 1) -> QPixmap:
        img = self.rgba(slice).copy()
        img = img.resize([img.size[0] * scaling, img.size[1] * scaling])
        qPixmap = QPixmap.fromImage(ImageQt.ImageQt(img))
        return qPixmap


class nslice:
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


class MouseTracker(QtCore.QObject):
    positionChanged = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self._widget.installEventFilter(self)

    @property
    def widget(self):
        return set._widget

    def eventFilter(self, o, e):
        # if o is self._widget and e.type() == QtCore.QEvent.Type.MouseMove:
        if o is self._widget and e.type() == QtCore.QEvent.Type.MouseButtonPress:
            self.positionChanged.emit(e.pos())
        return super().eventFilter(o, e)

    def lbl2npcoord(self, ypos: int, ysize: int, scaling: int):
        # Label coordinates to numpy indexes
        # y Axis is inverted for label coordinates
        new_pos = ysize * scaling - ypos
        return new_pos

    def np2lblcoord(
        self, xpos: int, ypos: int, xsize: int, ysize: int, scaling: int
    ) -> Tuple[int, int]:
        # numpy indexes to label coordinates
        new_x_pos = int(xpos / scaling)
        # y Axis is inverted for label coordinates
        new_y_pos = ysize - int(ypos / scaling)
        return new_x_pos, new_y_pos


def applyMask2Image(img: nifti_img, in_mask: nifti_img) -> nifti_img:
    if np.array_equal(img.size[0:2], in_mask.size[0:2]):
        # img_masked = nifti_img()
        img_masked = img.copy()
        mask = in_mask.copy()
        mask.array[mask.array == 0] = np.nan
        # mask.array = np.rot90(mask.array)
        for idx in range(img.size[3]):
            img_masked.array[:, :, :, idx] = np.multiply(
                img.array[:, :, :, idx], mask.array[:, :, :, 0]
            )
        return img_masked


def Signal2CSV(img: nifti_img, path: str | None = None):
    csvdata = dict()
    data = None
    for idx in range(img.size[0]):
        for idy in range(img.size[1]):
            for idz in range(img.size[2]):
                if not math.isnan(img.array[idx, idy, idz, 0]):
                    data = (
                        np.vstack((data, img.array[idx, idy, idz, :]))
                        if data is not None
                        else img.array[idx, idy, idz, :]
                    )
    with open(r"bvalues.bval", "r") as f:
        bvalues = list(str(x) for x in f.read().split("\n"))
    df = pd.DataFrame(data, columns=bvalues)
    if path is not None:
        df.to_excel(path, index=False)
    else:
        return df


def overlayImage(
    img: nifti_img,
    mask: nifti_img,
    slice: int = 0,
    alpha: int = 126,
    scaling: int = 2,
    color: str = "red",
) -> Image:
    if np.array_equal(img.size[0:3], mask.size[0:3]):
        _Img = img.rgba(slice).copy()
        imgMask = mask.rgba(slice).copy()
        imgOverlay = ImageOps.colorize(imgMask.convert("L"), black="black", white=color)
        alphamap = ImageOps.colorize(
            imgMask.convert("L"), black="black", white=(alpha, alpha, alpha)
        )
        imgOverlay.putalpha(alphamap.convert("L"))
        _Img.paste(imgOverlay, [0, 0], mask=imgOverlay)
        _Img = _Img.resize([_Img.size[0] * scaling, _Img.size[1] * scaling])
        # img_masked = img_masked.rotate(90)
        return _Img
