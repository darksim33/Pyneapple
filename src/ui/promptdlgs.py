import numpy as np
from PyQt6 import QtWidgets, QtGui

from src.utils import Nii, NiiSeg


class ReshapeSegDlg(QtWidgets.QDialog):
    def __init__(self, nii_img: Nii, nii_seg: NiiSeg):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_label = QtWidgets.QLabel()
        button_layout = QtWidgets.QHBoxLayout()
        self.close_bttn = QtWidgets.QPushButton()
        self.ok_bttn = QtWidgets.QPushButton()
        button_layout.addWidget(self.ok_bttn)
        button_layout.addSpacerItem(
            QtWidgets.QSpacerItem(
                28,
                28,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        )
        button_layout.addWidget(self.ok_bttn)
