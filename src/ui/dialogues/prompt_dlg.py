from __future__ import annotations
from typing import Callable
from PyQt6 import QtWidgets, QtCore
from scipy import ndimage

from src.utils import Nii, NiiSeg
from src.fit.parameters import Parameters, NNLSParams, IVIMParams, IDEALParams


class BasicPromptDlg(QtWidgets.QDialog):
    def __init__(
        self,
        title: str | None = None,
        text: str | None = None,
        accept_signal: Callable | QtCore.pyqtSignal | None = None,
        accept_button_txt: str | None = "Accept",
        close_button_txt: str | None = "Close",
    ):
        super().__init__()
        self._text = text
        self._title = title
        self.accept_signal = accept_signal
        self.accept_button_txt = accept_button_txt
        self.close_button_txt = close_button_txt
        self._initUI()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, string: str):
        self._text = string
        self.main_label.setText(string)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, string: str):
        self._title = string
        self.setWindowTitle(string)

    def _initUI(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
            )
        )
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_label = QtWidgets.QLabel()
        self.main_label.setText(self.text)
        self.main_layout.addWidget(self.main_label)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addSpacerItem(
            QtWidgets.QSpacerItem(
                28,
                28,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        )
        self.accept_button = QtWidgets.QPushButton()
        self.accept_button.setText(self.accept_button_txt)
        if self.accept_signal is not None:
            self.accept_button.clicked.connect(self.accept_signal)
        else:
            self.accept_button.clicked.connect(self.accept)
        button_layout.addWidget(self.accept_button)

        self.close_button = QtWidgets.QPushButton()
        self.close_button.setText(self.close_button_txt)
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        self.main_layout.addLayout(button_layout)
        self.setLayout(self.main_layout)

        self.setMinimumSize(self.main_label.sizeHint())


class ReshapeSegDlg(BasicPromptDlg):
    def __init__(self, img: Nii, seg: NiiSeg):
        super().__init__(
            title="Segmentation shape mismatch:",
            text="The shape of the segmentation does not match the image shape.\n"
            "Do you want to scale the segmentation shape to the image shape?",
            accept_signal=lambda: self.reshape(self.img, self.seg),
        )
        self.img = img
        self.seg = seg
        self.new_seg = None

    def reshape(self, *args):
        img: Nii = args[0]
        seg: NiiSeg = args[1]
        new_array = ndimage.zoom(
            seg.array[..., -1],
            (
                img.array.shape[0] / seg.array.shape[0],
                img.array.shape[1] / seg.array.shape[1],
                img.array.shape[2] / seg.array.shape[2],
            ),
            order=0,
        )
        print(f"Seg.shape from {seg.array.shape} to {new_array.shape}")
        self.new_seg = NiiSeg().from_array(new_array, seg.header, path=seg.path)
        self.accept()


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


class AlreadyLoadedSegDlg(BasicMessageBox):
    def __init__(self):
        super().__init__(
            title="Segmentation already loaded:",
            message="There is already a Segmentation loaded.\n"
            "Do you want to keep this segmentation?",
        )


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


class FitParametersMessageBox(BasicMessageBox):
    def __init__(self, fit_params: Parameters | IVIMParams | NNLSParams):
        title = "Parameter missmatch detected:"
        if isinstance(fit_params, IVIMParams):
            text = (
                "Currently IVIM parameters are loaded.\nDo you want to overwrite them?"
            )
        elif isinstance(fit_params, IDEALParams):
            text = (
                "Currently IDEAL parameters are loaded.\nDo you want to overwrite them?"
            )
        elif isinstance(fit_params, NNLSParams):
            text = (
                "Currently NNLS parameters are loaded.\nDo you want to overwrite them?"
            )
        else:
            text = "Test"
        super().__init__(
            title=title,
            message=text,
        )


class IDEALDimensionMessageBox(BasicMessageBox):
    def __init__(self):
        title = "Dimension Step missmatch detected:"
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
