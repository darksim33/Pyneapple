import sys
from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path
from utils import *
from PIL import ImageQt


class data:
    def __init__(self):
        self.nii_img: nifti_img = nifti_img()
        self.nii_mask: nifti_img = nifti_img()
        self.nii_img_masked: nifti_img = nifti_img()
        self.nii_dyn: nifti_img = nifti_img()
        self.plt = plt_settings()


class plt_settings:
    def __init__(self):
        self.nslice: int = nslice()
        self.scaling: int = 4
        self.overlay: bool = False
        self.alpha: int = 126


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        # Window setting
        self.setMinimumSize(512, 512)
        self.setWindowTitle("Pineapple")
        self.mainWidget = QtWidgets.QWidget()
        self.data = data()

        # Menubar
        self._createMenuBar()

        # Main vertical Layout
        self.main_hLayout = QtWidgets.QHBoxLayout()
        self.main_vLayout = QtWidgets.QVBoxLayout()
        self.main_AX = QtWidgets.QLabel()
        self.main_AX.setPixmap(
            QtGui.QPixmap(Path("ui", "resources", "image-not-available.jpg").__str__())
        )
        self.main_AX.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_vLayout.addWidget(self.main_AX)
        self.SliceHlayout = QtWidgets.QHBoxLayout()
        # Slider
        self.SliceSldr = QtWidgets.QSlider()
        self.SliceSldr.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.SliceSldr.setEnabled(False)
        self.SliceSldr.setMinimum(1)
        self.SliceSldr.setMaximum(100)
        self.data.plt.nslice.number = self.SliceSldr.value()
        self.SliceHlayout.addWidget(self.SliceSldr)
        # SpinBox
        self.SliceSpnBx = QtWidgets.QSpinBox()
        self.SliceSpnBx.setValue(1)
        self.SliceSpnBx.setEnabled(False)
        self.SliceHlayout.addWidget(self.SliceSpnBx)
        self.main_vLayout.addLayout(self.SliceHlayout)

        self.main_hLayout.addLayout(self.main_vLayout)
        self.mainWidget.setLayout(self.main_hLayout)

        self.setCentralWidget(self.mainWidget)

        # StatusBar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        # Connect Actions
        self._connectActions()

    def _createMenuBar(self):
        # Setup Menubar
        menuBar = self.menuBar()
        fileMenu = QtWidgets.QMenu("&File", self)
        self.loadImage = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "&Load Image...",
            self,
        )
        self.loadMask = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "&Load Mask...",
            self,
        )
        self.saveMaskedImage = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "&Save Masked Image...",
            self,
        )
        self.saveMaskedImage.setEnabled(False)
        fileMenu.addAction(self.loadImage)
        fileMenu.addAction(self.loadMask)
        fileMenu.addSeparator()
        fileMenu.addAction(self.saveMaskedImage)

        menuBar.addMenu(fileMenu)
        editMenu = QtWidgets.QMenu("&Edit", self)
        self.rotMask = QtGui.QAction("&Rotate Mask clockwise")
        self.rotMask.setEnabled(False)
        self.mask2img = QtGui.QAction("&Apply Mask to Image", self)
        self.mask2img.setEnabled(False)
        editMenu.addAction(self.mask2img)
        menuBar.addMenu(editMenu)

        viewMenu = QtWidgets.QMenu("&View", self)
        self.plt_overlay = QtGui.QAction("Show Mask Overlay", self)
        self.plt_overlay.setEnabled(False)
        self.plt_overlay.setCheckable(True)
        self.plt_overlay.setChecked(False)
        viewMenu.addAction(self.plt_overlay)
        self.plt_showMaskedImage = QtGui.QAction("Display Image with applied Mask")
        self.plt_showMaskedImage.setEnabled(False)
        self.plt_showMaskedImage.setCheckable(True)
        self.plt_showMaskedImage.setChecked(False)
        viewMenu.addAction(self.plt_showMaskedImage)
        menuBar.addMenu(viewMenu)

        evalMenu = QtWidgets.QMenu("&Evaluation", self)
        evalMenu.setEnabled(False)
        menuBar.addMenu(evalMenu)

    def _connectActions(self):
        self.loadImage.triggered.connect(self._loadImage)
        self.loadMask.triggered.connect(self._loadMask)
        self.saveMaskedImage.triggered.connect(self._saveMaskedImage)
        self.SliceSldr.valueChanged.connect(self._SliceSldrChanged)
        self.SliceSpnBx.valueChanged.connect(self._SliceSpnBxChanged)
        self.mask2img.triggered.connect(self._mask2img)
        self.plt_overlay.toggled.connect(self._plt_overlay)
        self.plt_showMaskedImage.toggled.connect(self._plt_showMaskedImage)

    def _loadImage(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self)[0]
        file = Path(path) if path else None
        self.data.nii_img = nifti_img(file)
        if self.data.nii_img.path is not None:
            self.data.plt.nslice.number = self.SliceSldr.value()
            self.SliceSldr.setEnabled(True)
            self.SliceSldr.setMaximum(self.data.nii_img.size[2])
            self.SliceSpnBx.setEnabled(True)
            self.SliceSpnBx.setMaximum(self.data.nii_img.size[2])
            self.plt_overlay.setEnabled(True)
            self.setupImage()
            self.mask2img.setEnabled(
                True
            ) if self.data.nii_mask.path else self.mask2img.setEnabled(False)

    def _loadMask(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self)[0]
        file = Path(path) if path else None
        self.data.nii_mask = nifti_img(file)
        if self.data.nii_mask:
            self.data.nii_mask.mask = True
            self.mask2img.setEnabled(
                True
            ) if self.data.nii_img.path else self.mask2img.setEnabled(False)

    def _saveMaskedImage(self):
        fname = self.data.nii_img.path
        fname = Path(str(fname).replace(fname.stem, fname.stem + "_masked"))
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Masked Image",
                fname.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.data.nii_img_masked.save(file)

    def _mask2img(self):
        self.data.nii_img_masked = applyMask2Image(
            self.data.nii_img, self.data.nii_mask
        )
        self.plt_showMaskedImage.setEnabled(True)
        self.saveMaskedImage.setEnabled(True)

    def _plt_overlay(self):
        self.data.plt.overlay = self.plt_overlay.isChecked()
        self.setupImage()

    def _plt_showMaskedImage(self):
        if self.plt_showMaskedImage.isChecked():
            self.plt_overlay.setChecked(False)
            self.plt_overlay.setEnabled(False)
            self.data.plt.overlay = False
            self.setupImage("masked")
        else:
            self.plt_overlay.setEnabled(True)
            self.setupImage()

    def _SliceSldrChanged(self):
        self.data.plt.nslice.number = self.SliceSldr.value()
        self.SliceSpnBx.setValue(self.SliceSldr.value())
        self.setupImage()

    def _SliceSpnBxChanged(self):
        self.data.plt.nslice.number = self.SliceSpnBx.value()
        self.SliceSldr.setValue(self.SliceSpnBx.value())
        self.setupImage()

    def setupImage(self, which: str = "img"):
        if which == "img":
            self.data.plt.nslice.number = self.SliceSldr.value()
            if (
                self.data.plt.overlay
                and self.data.nii_img.path
                and self.data.nii_mask.path
            ):
                img = overlayImage(
                    self.data.nii_img,
                    self.data.nii_mask,
                    self.data.plt.nslice.value,
                    self.data.plt.alpha,
                    self.data.plt.scaling,
                )
                qImg = QPixmap.fromImage(ImageQt.ImageQt(img))
            else:
                self.plt_overlay.setChecked(False)
                self.data.plt.overlay = False
                qImg = self.data.nii_img.nii2QPixmap(
                    self.data.plt.nslice.value, self.data.plt.scaling
                )
        elif which == "masked":
            qImg = self.data.nii_img_masked.nii2QPixmap(
                self.data.plt.nslice.value, self.data.plt.scaling
            )
        self.main_AX.setPixmap(qImg)
        self.setMinimumSize(qImg.size())


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()  # QtWidgets.QWidget()
window.show()
app.exec()
