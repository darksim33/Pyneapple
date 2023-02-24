import sys
# import pathlib
from scipy import ndimage
from PyQt6 import QtWidgets
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
import nibabel as nib

from ui_NNLSDynApp import Ui_MainWindow


class nifti_img:
    def __init__(self, path) -> None:
        nii = nib.load(path)
        nifti_img.array = nii.get_fdata()
        # nifti_img.array = np.array(nii.dataobj)
        nifti_img.affine = nii.affine
        nifti_img.header = nii.header
        nifti_img.size = nii.shape

    def nii2QPixmap(self, slice):
        scalingfactor = 3
        img = np.rot90(self.array[:, :, slice, 1])
        img_norm = (img - img.min()) / (img.max() - img.min())
        img_zoomed = ndimage.zoom(
            img_norm, (scalingfactor, scalingfactor), order=0, mode="nearest"
        )
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


def display_img(nii, axis):
    slice = 10
    axis.setPixmap(nii.nii2QPixmap(slice))


class appData:
    def __init__(self) -> None:
        self.plot_boundries = np.array([0.0001, 0.2])
        self.plot_slice = None
        self.imgDyn = np.array()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()

        self.appdata = appData

        self.setupUi(self)
        self.setWindowTitle("New NNLSDynApp")
        self.BttnImgDynLoad.clicked.connect(self.callback_BttnImgDynLoad)
        self.AXImgDyn.setPixmap(QPixmap("ui\\resources\\test.jpg"))
        # self.AXPltDyn.setPixmap(QPixmap("ui\\resources\\test.jpg"))

    def callback_BttnImgDynLoad(self):
        """fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
        )"""
        fname = "pat01_img.nii"
        self.appdata.imgDyn = nifti_img(fname)
        display_img(self.appdata.imgDyn, self.AXImgDyn)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
