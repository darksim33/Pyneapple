from PyQt6 import QtWidgets
from scipy import ndimage

from src.utils import Nii, NiiSeg


class ReshapeSegDlg(QtWidgets.QDialog):
    def __init__(self, img: Nii, seg: NiiSeg):
        super().__init__()
        self.img = img
        self.seg = seg
        self.new_seg = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Segmentation shape mismatch:")
        self.setWindowIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
            )
        )
        # self.setFixedSize(256, 96)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_label = QtWidgets.QLabel()
        self.main_label.setText(
            "The shape of the segmentation does not match the image shape.\n"
            "Do you want to scale the segmentation shape to the image shape?"
        )
        self.main_layout.addWidget(self.main_label)
        button_layout = QtWidgets.QHBoxLayout()
        self.accept_button = QtWidgets.QPushButton()
        self.accept_button.setText("Accept")
        self.accept_button.clicked.connect(lambda: self.reshape(self.img, self.seg))
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

    # @staticmethod
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
