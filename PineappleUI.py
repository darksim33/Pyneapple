import sys
from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path
from utils import *
from fitting import *
from plotting import plotting
from PIL import ImageQt
from multiprocessing import freeze_support
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# v0.3


class appData:
    def __init__(self):
        self.nii_img: nii = nii()
        self.nii_mask: nii_seg = nii_seg()
        # self.nii_seg: nii_seg = nii_seg()
        self.nii_img_masked: nii = nii()
        self.nii_dyn: nii = nii()
        self.plt = self._pltSettings()
        self.fit = self._fitData()

    class _pltSettings:
        def __init__(self):
            self.nslice: nslice = nslice(0)
            self.scaling: int = 4
            self.overlay: bool = False
            self.alpha: int = 126
            self.showPlt: bool = False
            self.qImg: QPixmap = None
            self.whichImg: str = "Img"
            self.pos = [10, 10]

    class _fitData:
        def __init__(self):
            self.NNLS = fitData("NNLS")
            self.NNLSreg = fitData("NNLSreg")
            self.mono = fitData("mono")
            self.mono_t1 = fitData("mono_t1")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: Path | str = None) -> None:
        super(MainWindow, self).__init__()
        self.data = appData()
        self.settings = QtCore.QSettings("MyApp", "Pyneapple")
        # initiate UI
        self._setupUI()
        # Connect Actions
        self._connectActions()

        # Load Settings
        # try:
        #     self._plt_show()
        # except:
        #     self.settings.setValue("plt_show", False)
        #     self._plt_show()
        self.settings.setValue("plt_show", False)
        # if function is UI ist initiated with a Path to a nifti load the nifti
        if path:
            self._loadImage(path)

    def _setupUI(self):
        # Window setting
        self.setMinimumSize(512, 512)
        self.setWindowTitle("Pineapple")
        img = Path(Path(__file__).parent, "resources", "Logo.png").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        self.mainWidget = QtWidgets.QWidget()

        # Menubar
        self._createMenuBar()

        # Context Menu
        self._createContextMenu()

        # Main vertical Layout
        self.main_hLayout = QtWidgets.QHBoxLayout()  # Main horzizontal Layout
        self.main_vLayout = QtWidgets.QVBoxLayout()  # Main Layout for img ans slider
        self.main_AX = QtWidgets.QLabel()

        self.data.plt.qImg = QtGui.QPixmap(
            # Path(Path(__file__).parent, "resources", "noImage.png").__str__()
            Path(Path(__file__).parent, "resources", "PyNeapple_BW_JJ.png").__str__()
        )
        self.main_AX.setPixmap(self.data.plt.qImg)
        self.main_AX.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # self.main_AX.setScaledContents(True)
        self.main_AX.installEventFilter(self)
        self.main_vLayout.addWidget(self.main_AX)
        self.SliceHlayout = QtWidgets.QHBoxLayout()  # Layout for Slider ans Spinbox
        # Slider
        self.SliceSldr = QtWidgets.QSlider()
        self.SliceSldr.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.SliceSldr.setEnabled(False)
        self.SliceSldr.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.SliceSldr.setTickInterval(1)
        self.SliceSldr.setMinimum(1)
        self.SliceSldr.setMaximum(20)
        self.SliceHlayout.addWidget(self.SliceSldr)
        # SpinBox
        self.SliceSpnBx = QtWidgets.QSpinBox()
        self.SliceSpnBx.setValue(1)
        self.SliceSpnBx.setEnabled(False)
        self.SliceSpnBx.setMinimumWidth(20)
        self.SliceSpnBx.setMaximumWidth(40)
        self.SliceHlayout.addWidget(self.SliceSpnBx)

        self.main_vLayout.addLayout(self.SliceHlayout)

        # Plotting Frame
        self.figure = Figure()
        self.figCanvas = FigureCanvas(self.figure)
        self.figAX = self.figure.add_subplot(111)
        self.figAX.set_xscale("log")
        self.figAX.set_xlabel("D (mm²/s)")

        self.main_hLayout.addLayout(self.main_vLayout)
        self.mainWidget.setLayout(self.main_hLayout)

        self.setCentralWidget(self.mainWidget)

        # StatusBar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

    def _createMenuBar(self):
        # Setup Menubar
        menuBar = self.menuBar()
        # File Menu
        fileMenu = QtWidgets.QMenu("&File", self)
        self.loadImage = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Image...",
            self,
        )
        self.loadSeg = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Segmentation...",
            self,
        )
        self.loadDyn = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Dynamic Image...",
            self,
        )
        self.saveImage = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "Save Image...",
            self,
        )
        self.saveFitImage = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "Save Fit to NifTi...",
            self,
        )
        self.saveFitImage.setEnabled(False)
        self.saveMaskedImage = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "Save Masked Image...",
            self,
        )
        self.saveMaskedImage.setEnabled(False)
        fileMenu.addAction(self.loadImage)
        fileMenu.addAction(self.loadSeg)
        fileMenu.addAction(self.loadDyn)
        fileMenu.addSeparator()
        fileMenu.addAction(self.saveImage)
        fileMenu.addAction(self.saveFitImage)
        fileMenu.addAction(self.saveMaskedImage)
        menuBar.addMenu(fileMenu)

        # Edit Menu
        editMenu = QtWidgets.QMenu("&Edit", self)
        MaskMenu = QtWidgets.QMenu("&Mask Tools", self)
        OrientationMenu = QtWidgets.QMenu("&Orientation", self)
        self.rotMask = QtGui.QAction("&Rotate Mask clockwise", self)
        self.rotMask.setEnabled(False)
        OrientationMenu.addAction(self.rotMask)
        self.maskFlipUpDown = QtGui.QAction("Flip Mask Up-Down", self)
        self.maskFlipUpDown.setEnabled(False)
        OrientationMenu.addAction(self.maskFlipUpDown)
        self.maskFlipLeftRight = QtGui.QAction("Flip Mask Left-Right", self)
        self.maskFlipLeftRight.setEnabled(False)
        OrientationMenu.addAction(self.maskFlipLeftRight)
        MaskMenu.addMenu(OrientationMenu)
        self.mask2img = QtGui.QAction("&Apply on Image", self)
        self.mask2img.setEnabled(False)
        MaskMenu.addAction(self.mask2img)
        editMenu.addMenu(MaskMenu)
        menuBar.addMenu(editMenu)

        # Fitting Menu
        fitMenu = QtWidgets.QMenu("&Fitting", self)
        fitMenu.setEnabled(True)
        nnlsMenu = QtWidgets.QMenu("NNLS", self)
        self.fit_NNLS = QtGui.QAction("NNLS", self)
        nnlsMenu.addAction(self.fit_NNLS)
        self.fit_NNLSreg = QtGui.QAction("NNLS with regularisation", self)
        self.fit_NNLSreg.setEnabled(False)
        nnlsMenu.addAction(self.fit_NNLSreg)
        fitMenu.addMenu(nnlsMenu)
        monoMenu = QtWidgets.QMenu("Mono Exponential", self)
        self.fit_mono = QtGui.QAction("Monoexponential", self)
        monoMenu.addAction(self.fit_mono)
        self.fit_mono_t1 = QtGui.QAction("Monoexponential with T1", self)
        monoMenu.addAction(self.fit_mono_t1)
        # monoMenu.setEnabled(False)
        fitMenu.addMenu(monoMenu)
        menuBar.addMenu(fitMenu)

        # View Menu
        viewMenu = QtWidgets.QMenu("&View", self)
        imageMenu = QtWidgets.QMenu("Switch Image", self)
        self.plt_showImg = QtGui.QAction("Image", self)
        # imageMenu.addAction(self.plt_showImg)
        self.plt_showMask = QtGui.QAction("Mask", self)
        # imageMenu.addAction(self.plt_showMask)
        self.plt_showMaskedImage = QtGui.QAction("Image with applied Mask")
        self.plt_showMaskedImage.setEnabled(False)
        self.plt_showMaskedImage.setCheckable(True)
        self.plt_showMaskedImage.setChecked(False)
        imageMenu.addAction(self.plt_showMaskedImage)
        self.plt_showDyn = QtGui.QAction("Dynamic", self)
        # imageMenu.addAction(self.plt_showDyn)
        viewMenu.addMenu(imageMenu)
        self.plt_show = QtGui.QAction("Show Plot")
        self.plt_show.setEnabled(True)
        self.plt_show.setCheckable(True)
        viewMenu.addAction(self.plt_show)
        viewMenu.addSeparator()
        self.plt_DispType_SingleVoxel = QtGui.QAction(
            "Show Single Voxel Spectrum", self
        )
        self.plt_DispType_SingleVoxel.setCheckable(True)
        self.plt_DispType_SingleVoxel.setChecked(True)
        self.settings.setValue("plt_disp_type", "single_voxel")
        viewMenu.addAction(self.plt_DispType_SingleVoxel)

        self.plt_DispType_SegSpectrum = QtGui.QAction(
            "Show Segmentation Spectrum", self
        )
        self.plt_DispType_SegSpectrum.setCheckable(True)
        viewMenu.addAction(self.plt_DispType_SegSpectrum)
        viewMenu.addSeparator()
        self.img_overlay = QtGui.QAction("Show Mask Overlay", self)
        self.img_overlay.setEnabled(False)
        self.img_overlay.setCheckable(True)
        self.img_overlay.setChecked(False)
        self.settings.setValue("img_disp_overlay", True)
        viewMenu.addAction(self.img_overlay)
        menuBar.addMenu(viewMenu)

        evalMenu = QtWidgets.QMenu("Evaluation", self)
        evalMenu.setEnabled(False)
        # menuBar.addMenu(evalMenu)

    def _createContextMenu(self):
        self.contextMenu = QtWidgets.QMenu(self)
        pltMenu = QtWidgets.QMenu("Plotting", self)
        pltMenu.addAction(self.plt_show)
        pltMenu.addSeparator()

        pltMenu.addAction(self.plt_DispType_SingleVoxel)

        pltMenu.addAction(self.plt_DispType_SegSpectrum)

        self.contextMenu.addMenu(pltMenu)

    def _connectActions(self):
        self.loadImage.triggered.connect(self._loadImage)
        self.loadSeg.triggered.connect(self._loadSeg)
        self.loadDyn.triggered.connect(self._loadDyn)
        self.saveImage.triggered.connect(self._saveImage)
        self.saveMaskedImage.triggered.connect(self._saveMaskedImage)
        self.saveFitImage.triggered.connect(self._saveFitImage)
        self.maskFlipUpDown.triggered.connect(self._maskFlipUpDown)
        self.maskFlipLeftRight.triggered.connect(self._maskFlipLeftRight)
        self.mask2img.triggered.connect(self._mask2img)
        self.fit_NNLS.triggered.connect(lambda x: self._fit_NNLS("NNLS"))
        self.fit_NNLSreg.triggered.connect(lambda x: self._fit_NNLS("NNLSreg"))
        self.fit_mono.triggered.connect(lambda x: self._fit_mono("mono"))
        self.fit_mono_t1.triggered.connect(lambda x: self._fit_mono("mono_t1"))
        self.plt_showImg.triggered.connect(lambda x: self._switchImage("Img"))
        self.plt_showMask.triggered.connect(lambda x: self._switchImage("Mask"))
        self.plt_showDyn.triggered.connect(lambda x: self._switchImage("Dyn"))
        self.plt_show.triggered.connect(self._plt_show)
        self.img_overlay.toggled.connect(self._img_overlay)
        self.plt_DispType_SingleVoxel.toggled.connect(
            lambda x: self._switchPlt("single_voxel")
        )
        self.plt_DispType_SegSpectrum.toggled.connect(
            lambda x: self._switchPlt("seg_spectrum")
        )
        self.plt_showMaskedImage.toggled.connect(self._plt_showMaskedImage)
        self.SliceSldr.valueChanged.connect(self._SliceSldrChanged)
        self.SliceSpnBx.valueChanged.connect(self._SliceSpnBxChanged)

    def eventFilter(self, source, event):
        if source == self.main_AX:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if self.data.nii_img.path:
                    self.data.plt.pos = plotting.lbl2np(
                        event.pos().x(),
                        event.pos().y(),
                        self.data.nii_img.size[1],
                        self.data.plt.scaling,
                    )
                    self.statusBar.showMessage(
                        "(%d, %d)" % (self.data.plt.pos[0], self.data.plt.pos[1])
                    )
                    if self.settings.value("plt_show", type=bool):
                        if (
                            self.settings.value("plt_disp_type", type=str)
                            == "single_voxel"
                        ):
                            plotting.show_PixelSpectrum(
                                self.figAX, self.figCanvas, self.data
                            )
                        elif (
                            self.settings.value("plt_disp_type", type=str)
                            == "seg_spectrum"
                        ):
                            plotting.show_SegSpectrum(
                                self.figAX, self.figCanvas, self.data, 0
                            )
                            print("test")

        return super().eventFilter(source, event)

    def contextMenuEvent(self, event):
        self.contextMenu.popup(QtGui.QCursor.pos())

    def _loadImage(self, path: Path | str = None):
        if not path:
            path = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Image", "", "NifTi (*.nii *.nii.gz)"
            )[0]
        file = Path(path) if path else None
        self.data.nii_img = nii(file)
        if self.data.nii_img.path is not None:
            self.data.plt.nslice.number = self.SliceSldr.value()
            self.SliceSldr.setEnabled(True)
            self.SliceSldr.setMaximum(self.data.nii_img.size[2])
            self.SliceSpnBx.setEnabled(True)
            self.SliceSpnBx.setMaximum(self.data.nii_img.size[2])
            self.settings.setValue("img_disp_type", "Img")
            self.setupImage()
            self.mask2img.setEnabled(True if self.data.nii_mask.path else False)
            self.img_overlay.setEnabled(True if self.data.nii_mask.path else False)

    def _loadSeg(self):
        path = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Mask Image", "", "NifTi (*.nii *.nii.gz)"
        )[0]
        file = Path(path) if path else None
        self.data.nii_mask = nii_seg(file)
        if self.data.nii_mask:
            self.data.nii_mask.mask = True
            self.mask2img.setEnabled(True if self.data.nii_mask.path else False)
            self.maskFlipUpDown.setEnabled(True)
            self.maskFlipLeftRight.setEnabled(True)

            self.img_overlay.setEnabled(True if self.data.nii_mask.path else False)
            self.img_overlay.setChecked(True if self.data.nii_mask.path else False)
            self.settings.setValue(
                "img_disp_overlay", True if self.data.nii_mask.path else False
            )

    def _loadDyn(self):
        path = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Dynamic Image", "", "NifTi (*.nii *.nii.gz)"
        )[0]
        file = Path(path) if path else None
        self.data.nii_dyn = nii(file)
        if self.settings.value("plt_show", type=bool):
            plotting.show_Spectrum(self.figAX, self.figCanvas, self.data)

    def _saveImage(self):
        fname = self.data.nii_img.path
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Image",
                fname.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.data.nii_img.save(file)

    def _saveFitImage(self):
        fname = self.data.nii_img.path
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Fit Image",
                fname.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.data.nii_dyn.save(file)

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

    def _maskFlipUpDown(self):
        # Images are rotated 90 degrees so lr and ud are switched
        self.data.nii_mask.array = np.fliplr(self.data.nii_mask.array)
        self.setupImage()

    def _maskFlipLeftRight(self):
        # Images are rotated 90 degrees so lr and ud are switched
        self.data.nii_mask.array = np.flipud(self.data.nii_mask.array)
        self.setupImage()

    def _mask2img(self):
        self.data.nii_img_masked = processing.mergeNiiImages(
            self.data.nii_img, self.data.nii_mask
        )
        if self.data.nii_img_masked:
            self.plt_showMaskedImage.setEnabled(True)
            self.saveMaskedImage.setEnabled(True)

    def _fit_NNLS(self, model: str):
        self.mainWidget.setCursor(QtCore.Qt.CursorShape.WaitCursor)

        if model == "NNLS":
            self.data.fit.NNLS.img = self.data.nii_img
            self.data.fit.NNLS.mask = self.data.nii_mask
            self.data.fit.NNLS.fitParams = NNLSParams(model, nbins=250)
        elif model == "NNLSreg":
            self.data.fit.NNLSreg.img = self.data.nii_img
            self.data.fit.NNLSreg.mask = self.data.nii_mask
            self.data.fit.NNLSreg.fitParams = NNLSParams("NNLSreg", nbins=250)
        self.data.nii_dyn = setupFitting(getattr(self.data.fit, model))

        self.saveFitImage.setEnabled(True)
        self.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def _fit_mono(self, model: str):
        self.mainWidget.setCursor(QtCore.Qt.CursorShape.WaitCursor)

        if model == "mono":
            self.data.fit.mono.img = self.data.nii_img
            self.data.fit.mono.mask = self.data.nii_mask
            self.data.fit.mono.fitParams = MonoParams("mono")
            self.data.fit.mono.fitParams.bValues = np.array(
                [
                    0,
                    5,
                    10,
                    20,
                    30,
                    40,
                    50,
                    75,
                    100,
                    150,
                    200,
                    250,
                    300,
                    400,
                    525,
                    750,
                ]
            )
        elif model == "mono_t1":
            self.data.fit.mono_t1.img = self.data.nii_img
            self.data.fit.mono_t1.mask = self.data.nii_mask
            self.data.fit.mono_t1.fitParams = MonoParams("mono_t1")
            self.data.fit.mono_t1.fitParams.bValues = np.array(
                [
                    0,
                    5,
                    10,
                    20,
                    30,
                    40,
                    50,
                    75,
                    100,
                    150,
                    200,
                    250,
                    300,
                    400,
                    525,
                    750,
                ]
            )
            self.data.fit.mono_t1.fitParams.variables.TM = (
                9.8  # add dynamic mixing times
            )
        self.data.nii_dyn = setupFitting(getattr(self.data.fit, model))

        self.saveFitImage.setEnabled(True)
        self.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def _switchImage(self, type: str = "Img"):
        self.settings.setValue("img_disp_type", type)
        self.setupImage()

    def _switchPlt(self, type: str = "single_voxel"):
        self.settings.setValue("plt_disp_type", type)
        if type == "single_voxel":
            if self.plt_DispType_SingleVoxel.isChecked():
                self.plt_DispType_SegSpectrum.setChecked(False)
            else:
                self.plt_DispType_SegSpectrum.setChecked(True)
        elif type == "seg_spectrum":
            if self.plt_DispType_SegSpectrum.isChecked():
                self.plt_DispType_SingleVoxel.setChecked(False)
            else:
                self.plt_DispType_SingleVoxel.setChecked(True)

    def _plt_show(self):
        if not self.plt_show.isChecked():
            self.figCanvas.setParent(None)
            self.figure.set_visible(False)
            self.settings.setValue("plt_show", False)
        else:
            self.main_hLayout.addWidget(self.figCanvas)
            self.settings.setValue("plt_show", True)
        self.resizeMainWindow()

    def _img_overlay(self):
        self.settings.setValue(
            "img_disp_overlay", True if self.img_overlay.isChecked() else False
        )
        self.setupImage()

    def _plt_showMaskedImage(self):
        if self.plt_showMaskedImage.isChecked():
            self.img_overlay.setChecked(False)
            self.img_overlay.setEnabled(False)
            self.settings.setValue("img_disp_overlay", False)
            self.setupImage()
        else:
            self.img_overlay.setEnabled(True)
            self.settings.setValue("img_disp_overlay", True)
            self.setupImage()

    def _SliceSldrChanged(self):
        self.data.plt.nslice.number = self.SliceSldr.value()
        self.SliceSpnBx.setValue(self.SliceSldr.value())
        self.setupImage()

    def _SliceSpnBxChanged(self):
        self.data.plt.nslice.number = self.SliceSpnBx.value()
        self.SliceSldr.setValue(self.SliceSpnBx.value())
        self.setupImage()

    def setupImage(self):
        nslice = self.data.plt.nslice.value
        scaling = self.data.plt.scaling
        if not self.plt_showMaskedImage.isChecked():
            self.data.plt.nslice.number = self.SliceSldr.value()
            if (
                self.settings.value("img_disp_overlay", type=bool)
                and self.data.nii_img.path
                and self.data.nii_mask.path
            ):
                img = plotting.overlayImage(
                    self.data.nii_img,
                    self.data.nii_mask,
                    nslice,
                    self.data.plt.alpha,
                    scaling,
                )
                if img:
                    self.data.plt.qImg = QPixmap.fromImage(ImageQt.ImageQt(img))
            else:
                self.img_overlay.setChecked(False)
                self.settings.setValue("img_disp_overlay", False)
                # self.data.plt.qImg = self.data.nii_img.QPixmap(nslice, scaling)
                self.data.plt.qImg = self.showImage().QPixmap(nslice, scaling)
        elif self.plt_showMaskedImage.isChecked():
            self.data.plt.qImg = self.data.nii_img_masked.QPixmap(nslice, scaling)
        if self.data.plt.qImg:
            self.main_AX.setPixmap(self.data.plt.qImg)
            # self.main_AX.setMinimumSize(self.data.plt.qImg.size())
            self.SliceSldr.setMaximumWidth(
                self.data.plt.qImg.width() - self.SliceSpnBx.maximumWidth()
            )
            self.main_AX.setFixedHeight(self.data.plt.qImg.height())
        self.resizeMainWindow()

    def showImage(self):
        if self.settings.value("img_disp_type") == "Img":
            return self.data.nii_img
        elif self.settings.value("img_disp_type") == "Mask":
            return self.data.nii_mask
        elif self.settings.value("img_disp_type") == "Seg":
            return self.data.nii_seg
        elif self.settings.value("img_disp_type") == "Dyn":
            return self.data.nii_dyn

        # if self.data.plt.whichImg == "Img":
        #     return self.data.nii_img
        # elif self.data.plt.whichImg == "Mask":
        #     return self.data.nii_mask
        # elif self.data.plt.whichImg == "Dyn":
        #     return self.data.nii_dyn

    def resizeMainWindow(self):
        self.main_hLayout.update()
        self.main_vLayout.update()
        if self.data.plt.qImg:
            if not self.settings.value("plt_show", type=bool):
                self.setMinimumHeight(
                    # self.main_AX.height()
                    self.data.plt.qImg.height()
                    + self.SliceHlayout.sizeHint().height()
                    + self.statusBar.height()
                    + 45  # MenuBar
                )
                self.setMinimumWidth(self.data.plt.qImg.width() + 50)
                self.resize(self.minimumWidth(), self.height())
            else:
                # width = self.main_AX.width()
                width = self.main_AX.width() + 450
                height = (
                    # self.main_AX.height()
                    self.data.plt.qImg.height()
                    + self.SliceHlayout.sizeHint().height()
                    + self.statusBar.height()
                    + 45
                )  # menuBar
                self.setMinimumSize(QtCore.QSize(width, height))
                # self.resize(self.minimumSize())
        else:
            self.setMinimumWidth(self.main_AX.width() * 2)


if __name__ == "__main__":
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()  # QtWidgets.QWidget()
    window.show()
    sys.exit(app.exec())


# # Start App, parse path if given
# argv = sys.argv[1:]
# if len(argv) > 0:
#     startPineappleUI(argv[0])
# else:
#     startPineappleUI()
