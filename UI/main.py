import sys

# import pathlib
from scipy import ndimage
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QPixmap, QImage, QIcon
import numpy as np
import nibabel as nib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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


class appData:
    def __init__(self):
        self.plt_boundries = np.array([0.0001, 0.2])
        self.plt_nslice = nslice()
        self.imgDyn = None


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


class MouseTracker(QtCore.QObject):
    positionChanged = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self._widget.setMouseTracking(True)
        self._widget.installEventFilter(self)

    @property
    def widget(self):
        return set._widget

    def eventFilter(self, o, e):
        if o is self._widget and e.type() == QtCore.QEvent.Type.MouseMove:
            self.positionChanged.emit(e.pos())
        return super().eventFilter(o, e)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        # Setup App
        self.setupUi(self)
        self.setWindowTitle("New NNLSDynApp")
        self.AXImgDyn.setPixmap(QPixmap("ui\\resources\\image-not-available.jpg"))
        self.BttnImgDynLoad.setIcon(QIcon("ui\\resources\\openFolder.png"))
        self.label_position = QtWidgets.QLabel(
            self.AXImgDyn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.label_position.setStyleSheet(
            "background-color: white; border: 1px solid black"
        )

        # Prepare Classes
        self.appdata = appData()
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        tracker = MouseTracker(self.AXImgDyn)
        tracker.positionChanged.connect(self.on_positionChanged)

        # Setup Callbacks
        self.BttnImgDynLoad.clicked.connect(self.callback_BttnImgDynLoad)
        self.spnBx_nSlice.valueChanged.connect(self.callback_spnBX_nSlice_changed)
        self.Sldr_nSlice.valueChanged.connect(self.callback_Sldr_nSlice_changed)
        self.AXImgDyn.setMouseTracking(True)

        # Setup Defaults
        # self.txtEdt_BoundMin.setText(self.appdata.plt_boundries[0])
        # self.txtEdt_BoundMax.setText(self.appdata.plt_boundries[1])

    def callback_BttnImgDynLoad(self):
        """fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
        )"""
        fname = "pat01_img_AmplDyn.nii"
        self.appdata.imgDyn = nifti_img(fname)
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        display_img(self.appdata.imgDyn, self.AXImgDyn, self.appdata.plt_nslice.value)
        self.spnBx_nSlice.setEnabled(True)
        self.spnBx_nSlice.setMaximum(self.appdata.imgDyn.size[2])
        self.Sldr_nSlice.setEnabled(True)
        self.Sldr_nSlice.setMaximum(self.appdata.imgDyn.size[2])
        self.AXImgDyn.setMouseTracking(True)

    def callback_spnBX_nSlice_changed(self):
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        self.Sldr_nSlice.setValue(self.spnBx_nSlice.value())
        display_img(self.appdata.imgDyn, self.AXImgDyn, self.appdata.plt_nslice.value)

    def callback_Sldr_nSlice_changed(self):
        self.appdata.plt_nslice.number = self.Sldr_nSlice.value()
        self.spnBx_nSlice.setValue(self.Sldr_nSlice.value())
        display_img(self.appdata.imgDyn, self.AXImgDyn, self.appdata.plt_nslice.value)

    def plt_spectrum(self):
        xdata = None
        ydata = None

        scene = QtWidgets.QGraphicsScene()
        self.AXPltDyn.setScene(scene)
        fig = Figure()
        axes = fig.gca()
        axes.plot(xdata, ydata)

    @QtCore.pyqtSlot(QtCore.QPoint)
    def on_positionChanged(self, pos):
        delta = QtCore.QPoint(30, -15)
        self.label_position.show()
        self.label_position.move(pos + delta)
        self.label_position.setText("(%d, %d)" % (pos.x(), pos.y()))
        self.label_position.adjustSize()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
