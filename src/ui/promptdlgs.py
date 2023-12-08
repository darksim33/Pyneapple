from __future__ import annotations
from typing import Callable
from PyQt6 import QtWidgets, QtCore
from scipy import ndimage

from src.utils import Nii, NiiSeg
from src.fit.parameters import Parameters, NNLSregParams, MultiExpParams


class BasicPromptDlg(QtWidgets.QDialog):
    def __init__(
        self,
        title: str | None = None,
        text: str | None = None,
        accept_signal: Callable | QtCore.pyqtSignal | None = None,
    ):
        super().__init__()
        self._text = text
        self._title = title
        self.accept_signal = accept_signal
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
        self.accept_button = QtWidgets.QPushButton()
        self.accept_button.setText("Accept")
        if self.accept_signal is not None:
            self.accept_button.clicked.connect(self.accept_signal)
        else:
            self.accept_button.clicked.connect(self.accept)
        button_layout.addWidget(self.accept_button)
        button_layout.addSpacerItem(
            QtWidgets.QSpacerItem(
                28,
                28,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        )
        self.close_button = QtWidgets.QPushButton()
        self.close_button.setText("Close")
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


class MissingSegDlg(BasicPromptDlg):
    def __init__(self):
        super().__init__(
            title="Missing Segmentation:",
            text="There is no Segmentation loaded at the moment.\n"
            "Do you want to fit every Pixel in the image?",
            accept_signal=None,
        )


class FitParametersDlg(BasicPromptDlg):
    def __init__(self, fit_params: Parameters | MultiExpParams | NNLSregParams):
        title = "Parameter missmatch detected:"
        text = ""
        if type(fit_params) == MultiExpParams:
            text = (
                "Currently IVIM parameters are loaded.\nDo you want to overwrite them?"
            )
        elif type(fit_params) == NNLSregParams:
            text = (
                "Currently NNLS parameters are loaded.\nDo you want to overwrite them?"
            )
        super().__init__(
            title=title,
            text=text,
            accept_signal=None,
        )
