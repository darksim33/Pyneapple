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


def display_img(nii, axis, slice):    
    axis.setPixmap(nii.nii2QPixmap(slice))


class appData():
    def __init__(self):
        self.plt_boundries = np.array([0.0001, 0.2])
        self.plt_nslice = nslice()
        self.imgDyn = None
        
class nslice():
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

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        # Setup App
        self.setupUi(self)
        self.setWindowTitle("New NNLSDynApp")
        self.AXImgDyn.setPixmap(QPixmap("ui\\resources\\image-not-available.jpg"))

        # Prepare Classes
        self.appdata = appData()
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        
        # Setup Callbacks
        self.BttnImgDynLoad.clicked.connect(self.callback_BttnImgDynLoad)
        self.spnBx_nSlice.valueChanged.connect(self.callback_spnBX_nSlice_changed)
        self.Sldr_nSlice.valueChanged.connect(self.callback_Sldr_nSlice_changed)

        # Setup Defaults        
        # self.txtEdt_BoundMin.setText(self.appdata.plt_boundries[0])
        # self.txtEdt_BoundMax.setText(self.appdata.plt_boundries[1])

    def callback_BttnImgDynLoad(self):
        """fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
        )"""
        fname = "pat01_img.nii"
        self.appdata.imgDyn = nifti_img(fname)
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        display_img(self.appdata.imgDyn, self.AXImgDyn, self.appdata.plt_nslice.value)
        self.spnBx_nSlice.setEnabled(True)
        self.spnBx_nSlice.setMaximum(self.appdata.imgDyn.size[2])
        self.Sldr_nSlice.setEnabled(True)
        self.Sldr_nSlice.setMaximum(self.appdata.imgDyn.size[2])
    
    def callback_spnBX_nSlice_changed(self):
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        self.Sldr_nSlice.setValue(self.spnBx_nSlice.value())
        display_img(self.appdata.imgDyn, self.AXImgDyn, self.appdata.plt_nslice.value)

    def callback_Sldr_nSlice_changed(self):
        self.appdata.plt_nslice.number = self.Sldr_nSlice.value()
        self.spnBx_nSlice.setValue(self.Sldr_nSlice.value())
        display_img(self.appdata.imgDyn, self.AXImgDyn, self.appdata.plt_nslice.value)



app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
