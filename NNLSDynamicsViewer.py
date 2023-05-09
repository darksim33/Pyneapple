import sys  # , os
import numpy as np
import nibabel as nib
from pathlib import Path
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QPixmap, QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image, ImageOps, ImageQt
from typing import Tuple
from UI import ui_NNLSDynApp
from copy import deepcopy


class nii:
    def __init__(self, path: str | Path | None = None) -> None:
        self.set_path(path)
        self.array = np.zeros((1, 1, 1, 1))
        self.affine = np.eye(4)
        self.header = np.array
        self.size = np.array
        self.mask: bool = False
        self.__load()

    def reset(self):
        self.__load()

    def save(self, name: str | Path, dtype: object = int):
        save_path = self.path.parent / name if self.path is not None else name
        # Save as Int/float
        array = np.array(self.array.astype(dtype).copy())
        header = self.header
        if dtype == int:
            header.set_data_dtype("i4")
        elif dtype == float:
            header.set_data_dtype("f4")
        new_nii = nib.Nifti1Image(
            array,
            self.affine,
            header,
        )
        # https://note.nkmk.me/en/python-numpy-dtype-astype/
        # https://brainder.org/2012/09/23/the-nifti-file-format/
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
        self.array = np.array(nii.get_fdata())
        while len(self.array.shape) < 4:
            self.array = np.expand_dims(self.array, axis=-1)
        self.affine = nii.affine
        self.header = nii.header
        self.size = np.array(self.array.shape)

    def copy(self):
        return deepcopy(self)

    def show(self, slice: int | None = None):
        img_rgb = self.rgba(slice)
        img_rgb.show()

    def fromArray(self, array: np.ndarray, ismask: bool = False):
        self.set_path = None
        self.array = array
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        self.size = array.shape
        self.mask = True if ismask else False
        return self

    def rgba(self, slice: int = 0, alpha: int = 1) -> Image:
        # Return RGBA PIL Image of nii slice
        # rot Image
        array = (
            np.rot90(self.array[:, :, slice, 0])
            if slice is not None
            else self.array[:, :, 0, 0]
        )
        # Add check for empty mask
        array_norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        # if nifti is mask -> Zeros get zero alpha
        alpha_map = array * alpha if self.mask else np.ones(array.shape)
        img_rgba = Image.fromarray(
            (np.dstack((array_norm, array_norm, array_norm, alpha_map)) * 255)
            .round()
            .astype(np.int8)  # Needed for Image
            .copy(),
            "RGBA",
        )

        return img_rgba

    def QPixmap(self, slice: int = 0, scaling: int = 1) -> QPixmap:
        if self.path:
            img = self.rgba(slice).copy()
            img = img.resize(
                [img.size[0] * scaling, img.size[1] * scaling], Image.NEAREST
            )
            qPixmap = QPixmap.fromImage(ImageQt.ImageQt(img))
            return qPixmap
        else:
            return None


class appData:
    def __init__(self):
        self.plt_boundries: np.ndarray = np.array([0.0001, 0.2])
        self.plt_nslice: nslice = nslice(0)
        self.plt_scaling: int = 2
        self.imgMain: nii = nii()
        self.imgDyn: nii = nii()


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


class MainWindow(QtWidgets.QMainWindow, ui_NNLSDynApp.Ui_MainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        # Setup App
        self.setupUi(self)
        self.setWindowTitle("NNLSDynApp")
        self.AXImgDyn.setPixmap(
            QPixmap(Path("ui", "resources", "image-not-available.jpg").__str__())
        )
        self.BttnImgDynLoad.setIcon(
            QIcon(Path("ui", "resources", "openFolder.png").__str__())
        )
        self.menubar.setEnabled(False)

        # Figure Initiation
        figPltDyn = Figure()
        self.figPltDynCanvas = FigureCanvas(figPltDyn)
        self.AXPltDyn = figPltDyn.add_subplot(111)
        self.hLayout_Main.addWidget(self.figPltDynCanvas)

        # Prepare Classes
        self.appdata = appData()
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        self.MouseTracker = MouseTracker(self.AXImgDyn)
        self.MouseTracker.positionChanged.connect(self.on_positionChanged)

        # Setup Callbacks
        self.BttnImgDynLoad.clicked.connect(self.callback_BttnImgDynLoad)
        self.spnBx_nSlice.valueChanged.connect(self.callback_spnBX_nSlice_changed)
        self.Sldr_nSlice.valueChanged.connect(self.callback_Sldr_nSlice_changed)

    def callback_BttnImgDynLoad(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
        )[0]
        # fname = "pat01_img_AmplDyn.nii"
        self.appdata.imgDyn.load(fname)
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        self.AXImgDyn.setPixmap(
            self.appdata.imgDyn.QPixmap(
                self.appdata.plt_nslice.value, self.appdata.plt_scaling
            )
        )
        # Setup Slice in UI
        self.spnBx_nSlice.setEnabled(True)
        self.spnBx_nSlice.setMaximum(self.appdata.imgDyn.size[2])
        self.Sldr_nSlice.setEnabled(True)
        self.Sldr_nSlice.setMaximum(self.appdata.imgDyn.size[2])
        self.AXImgDyn.setMouseTracking(True)
        # Scale AXImgDyn
        self.AXImgDyn.setMaximumSize(
            self.appdata.imgDyn.size[0] * self.appdata.plt_scaling,
            self.appdata.imgDyn.size[1] * self.appdata.plt_scaling,
        )
        self.AXImgDyn.setMinimumSize(
            self.appdata.imgDyn.size[0] * self.appdata.plt_scaling,
            self.appdata.imgDyn.size[1] * self.appdata.plt_scaling,
        )

    def callback_spnBX_nSlice_changed(self):
        self.appdata.plt_nslice.number = self.spnBx_nSlice.value()
        self.Sldr_nSlice.setValue(self.spnBx_nSlice.value())
        self.AXImgDyn.setPixmap(
            self.appdata.imgDyn.QPixmap(
                self.appdata.plt_nslice.value, self.appdata.plt_scaling
            )
        )

    def callback_Sldr_nSlice_changed(self):
        self.appdata.plt_nslice.number = self.Sldr_nSlice.value()
        self.spnBx_nSlice.setValue(self.Sldr_nSlice.value())
        self.AXImgDyn.setPixmap(
            self.appdata.imgDyn.QPixmap(
                self.appdata.plt_nslice.value, self.appdata.plt_scaling
            )
        )

    @QtCore.pyqtSlot(QtCore.QPoint)
    def on_positionChanged(self, pos):
        ypos = self.MouseTracker.lbl2npcoord(
            pos.y(), self.appdata.imgDyn.size[1], self.appdata.plt_scaling
        )
        msg = "(%d, %d)" % (pos.x(), ypos)
        self.statusbar.showMessage(msg)
        try:
            self.plt_spectrum(pos)
        except IndexError:
            pass

    def plt_spectrum(self, pos):
        x, y = self.MouseTracker.np2lblcoord(
            pos.x(),
            pos.y(),
            self.appdata.imgDyn.size[0],
            self.appdata.imgDyn.size[1],
            self.appdata.plt_scaling,
        )
        ydata = self.appdata.imgDyn.array[
            x,
            y,
            self.appdata.plt_nslice.value,
            :,
        ]
        nbins = np.shape(ydata)
        xdata = np.logspace(0.0001, 0.2, num=nbins[0])
        self.AXPltDyn.clear()
        self.AXPltDyn.plot(xdata, ydata)
        self.figPltDynCanvas.draw()


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


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
