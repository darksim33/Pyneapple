from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6 import QtWidgets
from scipy import ndimage

from pyneapple.utils.nifti import Nii, NiiSeg
from src.pyneapple.fit import Parameters, NNLSbaseParams, IVIMParams, IDEALParams

if TYPE_CHECKING:
    pass


class BasicMessageBox(QtWidgets.QMessageBox):
    def __init__(
        self, title: str, message: str, info_text: str | None = None, **kwargs
    ):
        super().__init__()

        self.setWindowTitle(title)
        self.setText(message)
        if info_text is not None:
            self.setInformativeText(info_text)
        self.setWindowIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
            )
        )
        self.setIcon(kwargs.get("icon", QtWidgets.QMessageBox.Icon.Warning))

        self.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
        )


class AlreadyLoadedSegMessageBox(BasicMessageBox):
    def __init__(self):
        super().__init__(
            title="Segmentation already loaded:",
            message="There is already a Segmentation loaded.\n"
            "Do you want to keep this segmentation?",
        )


class ReshapeSegMessageBox(BasicMessageBox):
    def __init__(self):
        super().__init__(
            title="Segmentation Shape Mismatch:",
            message="The shape of the segmentation does not match the image shape.",
            info_text="Do you want to scale the segmentation shape to the image shape?",
        )

    @staticmethod
    def reshape(*args):
        img: Nii = args[0]
        seg: NiiSeg = args[1]
        new_array = ndimage.zoom(
            seg.array,
            (
                img.array.shape[0] / seg.array.shape[0],
                img.array.shape[1] / seg.array.shape[1],
                img.array.shape[2] / seg.array.shape[2],
                1,
            ),
            order=0,
        )
        print(f"Seg.shape from {seg.array.shape} to {new_array.shape}")
        return NiiSeg().from_array(new_array, seg.header, path=seg.path)


class MissingSegmentationMessageBox(BasicMessageBox):
    def __init__(self):
        super().__init__(
            title="Missing Segmentation:",
            message="There is no Segmentation loaded at the moment.\n"
            "Do you want to fit every Pixel in the image?",
        )


class StillLoadedSegMessageBox(BasicMessageBox):
    def __init__(self):
        super().__init__(
            title="Segmentation still loaded:",
            message="Another Segmentation is still loaded.\n"
            "Do you want to keep this segmentation?",
        )


class ZeroPaddingMissmatchMessageBox(BasicMessageBox):
    def __init__(self):
        super().__init__(
            title="Zero-Padding Error:",
            message="Segmentation of same shape loaded.",
            info_text="The loaded Segmentation has the same shape as the loaded image.\nDo you want to perform Zero-padding on both?",
        )


class FitParametersMessageBox(BasicMessageBox):
    def __init__(self, fit_params: Parameters | IVIMParams | NNLSbaseParams):
        title = "Parameter missmatch:"
        if isinstance(fit_params, IVIMParams):
            text = (
                "Currently IVIM parameters are loaded.\nDo you want to overwrite them?"
            )
        elif isinstance(fit_params, IDEALParams):
            text = (
                "Currently IDEAL parameters are loaded.\nDo you want to overwrite them?"
            )
        elif isinstance(fit_params, NNLSbaseParams):
            text = (
                "Currently NNLS parameters are loaded.\nDo you want to overwrite them?"
            )
        else:
            text = "Unknown parameters detected.\nDo you want to overwrite them?"
        super().__init__(
            title=title,
            message=text,
        )


class IDEALSquarePlaneMessageBox(BasicMessageBox):
    def __init__(self):
        super().__init__(
            title="Matrix Shape Error",
            message="The Image Matrix is not square!",
            info_text="The Image Matrix should be square to perform proper IDEAL fitting.\n"
            "Do you want to make the image and the segmentation square?",
        )


class IDEALFinalDimensionStepMessageBox(BasicMessageBox):
    def __init__(self):
        title = "Dimension Step missmatch detected"
        text = (
            "Dimension of Image does not match final dimension step of IDEAL!\n"
            "The last step will be replaced by the image dimensions!"
        )
        super().__init__(title=title, message=text)


class RepeatedFitMessageBox(BasicMessageBox):
    def __init__(self) -> None:
        self.title = "Repeated Fit detected"
        self.message = "Found a already processed fit!"
        self.info_text = (
            "Do you want to aboard or continue and discard the previous fit results?"
        )

        super().__init__(self.title, self.message, self.info_text)

        self.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        self.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Abort
            | QtWidgets.QMessageBox.StandardButton.Discard
        )
