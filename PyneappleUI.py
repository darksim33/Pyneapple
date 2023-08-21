import sys, os

from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path

# from PyQt6.QtWidgets import QWidget
# from PIL import ImageQt
from typing import Callable
from multiprocessing import freeze_support
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.utils import *
import src.plotting as plotting
from src.fit import fit, parameters, model
from src.ui.fittingdlg import FittingWindow, FittingWidgets, FittingDictionaries
from src.ui.settingsdlg import SettingsWindow

# v0.4.2


class AppData:
    def __init__(self):
        self.nii_img: Nii = Nii()
        self.nii_seg: NiiSeg = NiiSeg()
        self.nii_img_masked: Nii = Nii()
        self.nii_dyn: Nii = Nii()
        self.plt = self._PltSettings()
        self.fit = self._FitData()

    class _PltSettings:
        def __init__(self):
            self.nslice: NSlice = NSlice(0)
            self.alpha: float = 0.5
            self.mask_patches = None

    class _FitData:
        def __init__(self):
            self.fit_data = fit.FitData()
            # self.NNLS = fit.FitData("NNLSreg")
            # self.NNLSreg = fit.FitData("NNLSreg")
            # self.NNLSregCV = fit.FitData("NNLSregCV")
            # self.mono = fit.FitData("mono")
            # self.mono_t1 = fit.FitData("mono_T1")
            # self.multiexp = fit.FitData("multiexp")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: Path | str = None) -> None:
        super(MainWindow, self).__init__()
        self.data = AppData()

        # Load Settings
        self._load_settings()

        # initiate UI
        self._setup_ui()

        self.fit_dlg = FittingWindow
        # if path:
        #     self._load_image(path)

    def _load_settings(self):
        self.settings = QtCore.QSettings("MyApp", "Pyneapple")
        if self.settings.value("last_dir", "") == "":
            self.settings.setValue("last_dir", os.path.abspath(__file__))
            self.settings.setValue("theme", "Light")  # "Dark", "Light"
        self.settings.setValue("plt_show", False)

    def _setup_ui(self):
        # ----- Window setting
        self.setMinimumSize(512, 512)
        self.setWindowTitle("Pyneapple")
        img = Path(Path(__file__).parent, "resources", "Logo.png").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        self.mainWidget = QtWidgets.QWidget()

        # ----- Menubar
        self._create_menu_bar()

        # ----- Context Menu
        self._createContextMenu()

        # ----- Main vertical Layout
        self.main_hLayout = QtWidgets.QHBoxLayout()  # Main horizontal Layout
        self.main_vLayout = QtWidgets.QVBoxLayout()  # Main Layout for img and slider

        # ----- Main Image Axis
        self.img_fig = Figure()
        self.img_canvas = FigureCanvas(self.img_fig)
        self.img_canvas.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
        )
        self.img_ax = self.img_fig.add_subplot(111)
        self.img_ax.axis("off")
        self.main_vLayout.addWidget(self.img_canvas)
        self.img_fig.canvas.mpl_connect("button_press_event", self.event_filter)

        self.img_ax.clear()

        if self.settings.value("theme") == "Dark":
            self.img_ax.imshow(
                np.array(
                    Image.open(
                        # Path(Path(__file__).parent, "resources", "noImage_white.png")
                        Path(
                            Path(__file__).parent, "resources", "PyneAppleLogo_gray.png"
                        )
                    )
                ),
                cmap="gray",
            )
            self.img_fig.set_facecolor("black")
        elif self.settings.value("theme") == "Light":
            self.img_ax.imshow(
                np.array(
                    Image.open(
                        Path(Path(__file__).parent, "resources", "noImage.png")
                        # Path(Path(__file__).parent, "resources", "PyNeapple_BW_JJ.png")
                    )
                ),
                cmap="gray",
            )
        self.img_ax.axis("off")

        # ----- Slider
        def _slice_sldr_changed(self):
            """Slice Slider Callback"""
            self.data.plt.nslice.number = self.SliceSldr.value()
            self.SliceSpnBx.setValue(self.SliceSldr.value())
            self.setup_image()

        self.SliceHLayout = QtWidgets.QHBoxLayout()  # Layout for Slider ans Spinbox
        self.SliceSldr = QtWidgets.QSlider()
        self.SliceSldr.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.SliceSldr.setEnabled(False)
        self.SliceSldr.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.SliceSldr.setTickInterval(1)
        self.SliceSldr.setMinimum(1)
        self.SliceSldr.setMaximum(20)
        self.SliceSldr.valueChanged.connect(lambda x: _slice_sldr_changed(self))
        self.SliceHLayout.addWidget(self.SliceSldr)

        # ----- SpinBox
        def _slice_spn_bx_changed(self):
            """Slice Spinbox Callback"""
            self.data.plt.nslice.number = self.SliceSpnBx.value()
            self.SliceSldr.setValue(self.SliceSpnBx.value())
            self.setup_image()

        self.SliceSpnBx = QtWidgets.QSpinBox()
        self.SliceSpnBx.setValue(1)
        self.SliceSpnBx.setEnabled(False)
        self.SliceSpnBx.setMinimumWidth(20)
        self.SliceSpnBx.setMaximumWidth(40)
        self.SliceSpnBx.valueChanged.connect(lambda x: _slice_spn_bx_changed(self))
        self.SliceHLayout.addWidget(self.SliceSpnBx)

        self.main_vLayout.addLayout(self.SliceHLayout)

        # Adjust Canvas and Slider size according to main window
        self._resize_canvas_size()
        self._resize_figure_axis()
        self.img_canvas.draw()

        # ----- Plotting Frame
        self.plt_vLayout = QtWidgets.QVBoxLayout()  # Layout for plots

        # ----- Signal
        self.plt_signal_fig = Figure()
        self.plt_signal_canvas = FigureCanvas(self.plt_signal_fig)
        self.plt_signal_AX = self.plt_signal_fig.add_subplot(111)
        self.plt_vLayout.addWidget(self.plt_signal_canvas)
        self.plt_signal_AX.set_xlabel("b-Values")

        # ----- Spectrum
        self.plt_spectrum_fig = Figure()
        self.plt_spectrum_canvas = FigureCanvas(self.plt_spectrum_fig)
        self.plt_spectrum_AX = self.plt_spectrum_fig.add_subplot(111)
        self.plt_spectrum_AX.set_xscale("log")
        self.plt_spectrum_AX.set_xlabel("D (mm²/s)")

        self.plt_vLayout.addWidget(self.plt_spectrum_canvas)

        self.main_hLayout.addLayout(self.main_vLayout)
        self.mainWidget.setLayout(self.main_hLayout)

        self.setCentralWidget(self.mainWidget)

        # ----- StatusBar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

    def _create_menu_bar(self):
        # ----- Setup Menubar

        menu_bar = self.menuBar()

        # ----- File Menu

        file_menu = QtWidgets.QMenu("&File", self)

        # Load Image
        def _load_image(self, path: Path | str = None):
            if not path:
                path = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    caption="Open Image",
                    directory="",
                    filter="NifTi (*.nii *.nii.gz)",
                )[0]
            if path:
                file = Path(path) if path else None
                self.data.nii_img = Nii(file)
                if self.data.nii_img.path is not None:
                    self.data.plt.nslice.number = self.SliceSldr.value()
                    self.SliceSldr.setEnabled(True)
                    self.SliceSldr.setMaximum(self.data.nii_img.array.shape[2])
                    self.SliceSpnBx.setEnabled(True)
                    self.SliceSpnBx.setMaximum(self.data.nii_img.array.shape[2])
                    self.settings.setValue("img_disp_type", "Img")
                    self.setup_image()
                    self.mask2img.setEnabled(True if self.data.nii_seg.path else False)
                    self.img_overlay.setEnabled(
                        True if self.data.nii_seg.path else False
                    )
                else:
                    print("Warning no file selcted")

        self.loadImage = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Image...",
            self,
        )
        self.loadImage.triggered.connect(lambda x: _load_image(self))
        file_menu.addAction(self.loadImage)

        # Load Segmentation
        def _load_seg(self):
            # TODO create overlay in advance before loading not on the fly
            path = QtWidgets.QFileDialog.getOpenFileName(
                self,
                caption="Open Mask Image",
                directory="",
                filter="NifTi (*.nii *.nii.gz)",
            )[0]
            if path:
                file = Path(path)
                self.data.nii_seg = NiiSeg(file)
                if self.data.nii_seg:
                    self.data.nii_seg.mask = True
                    self.mask2img.setEnabled(True if self.data.nii_seg.path else False)
                    self.maskFlipUpDown.setEnabled(True)
                    self.maskFlipLeftRight.setEnabled(True)
                    self.maskFlipBackForth.setEnabled(True)

                    self.img_overlay.setEnabled(
                        True if self.data.nii_seg.path else False
                    )
                    self.img_overlay.setChecked(
                        True if self.data.nii_seg.path else False
                    )
                    self.settings.setValue(
                        "img_disp_overlay", True if self.data.nii_seg.path else False
                    )
            else:
                print("Warning no file selcted")

        self.loadSeg = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Segmentation...",
            self,
        )
        self.loadSeg.triggered.connect(lambda x: _load_seg(self))
        file_menu.addAction(self.loadSeg)

        # Load dynamic Image
        def _load_dyn(self):
            path = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Dynamic Image", "", "NifTi (*.nii *.nii.gz)"
            )[0]
            if path:
                file = Path(path) if path else None
                self.data.nii_dyn = Nii(file)
            # if self.settings.value("plt_show", type=bool):
            #     Plotting.show_pixel_spectrum(self.plt_AX, self.plt_canvas, self.data)
            else:
                print("Warning no file selected")

        self.loadDyn = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Dynamic Image...",
            self,
        )
        self.loadDyn.triggered.connect(lambda x: _load_dyn(self))
        file_menu.addAction(self.loadDyn)
        file_menu.addSeparator()

        # Save Image
        def _save_image(self):
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

        self.saveImage = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "Save Image...",
            self,
        )
        self.saveImage.triggered.connect(lambda x: _save_image(self))
        file_menu.addAction(self.saveImage)

        # Save Fit Image
        def _save_fit_image(self):
            file_name = self.data.nii_img.path
            file = Path(
                QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save Fit Image",
                    file_name.__str__(),
                    "NifTi (*.nii *.nii.gz)",
                )[0]
            )
            self.data.nii_dyn.save(file)

        self.saveFitImage = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "Save Fit to NifTi...",
            self,
        )
        self.saveFitImage.setEnabled(False)
        self.saveFitImage.triggered.connect(lambda x: _save_fit_image(self))
        file_menu.addAction(self.saveFitImage)

        # Save masked image
        def _save_masked_image(self):
            file_name = self.data.nii_img.path
            file_name = Path(
                str(file_name).replace(file_name.stem, file_name.stem + "_masked")
            )
            file = Path(
                QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save Masked Image",
                    file_name.__str__(),
                    "NifTi (*.nii *.nii.gz)",
                )[0]
            )
            self.data.nii_img_masked.save(file)

        self.saveMaskedImage = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "Save Masked Image...",
            self,
        )
        self.saveMaskedImage.setEnabled(False)
        self.saveMaskedImage.triggered.connect(lambda x: _save_masked_image(self))
        file_menu.addAction(self.saveMaskedImage)

        file_menu.addSeparator()

        # Open Settings
        def _open_settings_dlg(self):
            self.settings_dlg = SettingsWindow(self, self.settings)
            self.settings_dlg.show()

        self.open_settings_dlg = QtGui.QAction(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_TitleBarMenuButton
            ),
            "Settings...",
            self,
        )
        self.open_settings_dlg.setEnabled(True)
        self.open_settings_dlg.triggered.connect(lambda x: _open_settings_dlg(self))
        file_menu.addAction(self.open_settings_dlg)

        menu_bar.addMenu(file_menu)

        # ----- Edit Menu
        edit_menu = QtWidgets.QMenu("&Edit", self)
        mask_menu = QtWidgets.QMenu("&Mask Tools", self)

        orientation_menu = QtWidgets.QMenu("&Orientation", self)
        self.rotMask = QtGui.QAction("&Rotate Mask clockwise", self)
        self.rotMask.setEnabled(False)
        orientation_menu.addAction(self.rotMask)

        # Flip Mask Up Down
        def _mask_flip_up_down(self):
            # Images are rotated 90 degrees so lr and ud are switched
            self.data.nii_seg.array = np.fliplr(self.data.nii_seg.array)
            self.data.nii_seg.calculate_polygons()
            self.setup_image()

        self.maskFlipUpDown = QtGui.QAction("Flip Mask Up-Down", self)
        self.maskFlipUpDown.setEnabled(False)
        self.maskFlipUpDown.triggered.connect(lambda x: _mask_flip_up_down(self))
        orientation_menu.addAction(self.maskFlipUpDown)

        # Flip Left Right
        def _mask_flip_left_right(self):
            # Images are rotated 90 degrees so lr and ud are switched
            self.data.nii_seg.array = np.flipud(self.data.nii_seg.array)
            self.data.nii_seg.calculate_polygons()
            self.setup_image()

        self.maskFlipLeftRight = QtGui.QAction("Flip Mask Left-Right", self)
        self.maskFlipLeftRight.setEnabled(False)
        self.maskFlipLeftRight.triggered.connect(lambda x: _mask_flip_left_right(self))
        orientation_menu.addAction(self.maskFlipLeftRight)

        # Flip Back Forth
        def _mask_flip_back_forth(self):
            self.data.nii_seg.array = np.flip(self.data.nii_seg.array, axis=2)
            self.data.nii_seg.calculate_polygons()
            self.setup_image()

        self.maskFlipBackForth = QtGui.QAction("Flip Mask Back-Forth", self)
        self.maskFlipBackForth.setEnabled(False)
        self.maskFlipBackForth.triggered.connect(lambda x: _mask_flip_back_forth(self))
        orientation_menu.addAction(self.maskFlipBackForth)

        mask_menu.addMenu(orientation_menu)

        # Mask to Image
        def _mask2img(self):
            self.data.nii_img_masked = Processing.merge_nii_images(
                self.data.nii_img, self.data.nii_seg
            )
            if self.data.nii_img_masked:
                self.plt_showMaskedImage.setEnabled(True)
                self.saveMaskedImage.setEnabled(True)

        self.mask2img = QtGui.QAction("&Apply on Image", self)
        self.mask2img.setEnabled(False)
        self.mask2img.triggered.connect(lambda x: _mask2img(self))
        mask_menu.addAction(self.mask2img)

        edit_menu.addMenu(mask_menu)
        menu_bar.addMenu(edit_menu)

        # ----- Fitting Procedure
        def _fit(self, model_name: str):
            fit_data = self.data.fit.fit_data
            dlg_dict = dict()
            if model_name in ("NNLS", "NNLSreg"):
                fit_data.fit_params = parameters.NNLSRegParams()
                fit_data.model_name = "NNLS"
                # Prepare Dlg Dict
                dlg_dict = FittingDictionaries.get_nnls_dict(fit_data)

            if model_name in ("mono", "mono_t1"):
                # if model_name in "mono":
                #     fit_data = self.data.fit.mono
                # elif model_name in "mono_t1":
                #     fit_data = self.data.fit.mono_t1
                if model_name in "mono":
                    fit_data.fit_params = parameters.MonoParams()
                    fit_data.model_name = "mono"
                elif model_name in "mono_t1":
                    fit_data.fit_params = parameters.MonoT1Params()
                    fit_data.model_name = "mono_t1"
                dlg_dict = FittingDictionaries.get_mono_dict(fit_data)
                if model_name in "mono_t1":
                    dlg_dict["TM"] = FittingWidgets.EditField(
                        "Mixing Time (TM)",
                        fit_data.fit_params.TM,
                        None,
                        "Set Mixing Time if you want to perform advanced Fitting",
                    )
            if model_name in "multiexp":
                # fit_data = self.data.fit.multiexp
                fit_data.fit_params = parameters.MultiTest()
                fit_data.model_name = "multiexp"
                dlg_dict = FittingDictionaries.get_multiexp_dict(fit_data)

            dlg_dict["b_values"] = FittingWidgets.PushButton(
                "Load B-Values",
                str(fit_data.fit_params.b_values),
                self._load_b_values,
                "Open File",
            )

            # Launch Dlg
            self.fit_dlg = FittingWindow(model_name, dlg_dict)
            self.fit_dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self.fit_dlg.exec()

            # Extract Parameters from dlg dict
            fit_data.fit_params.b_values = self._b_values_from_dict()
            self.fit_dlg.dict_to_attributes(fit_data.fit_params)

            if self.fit_dlg.run:
                if (
                    hasattr(fit_data.fit_params, "reg_order")
                    and fit_data.fit_params.reg_order == "CV"
                ):
                    fit_data.fit_params.model = model.Model.NNLS_reg_CV
                elif (
                    hasattr(fit_data.fit_params, "reg_order")
                    and fit_data.fit_params.reg_order != "CV"
                ):
                    fit_data.fit_params.reg_order = int(fit_data.fit_params.reg_order)

                self.mainWidget.setCursor(QtCore.Qt.CursorShape.WaitCursor)

                # Prepare Data
                fit_data.img = self.data.nii_img
                fit_data.seg = self.data.nii_seg

                if fit_data.fit_params.fit_area == "Pixel":
                    fit_data.fit_pixel_wise()
                    self.data.nii_dyn = Nii().from_array(
                        self.data.fit.fit_data.fit_results.spectrum
                    )

                elif fit_data.fit_area == "Segmentation":
                    fit_data.fit_segmentation_wise()

                self.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

                self.saveFitImage.setEnabled(True)

        # ----- Fitting Menu
        fit_menu = QtWidgets.QMenu("&Fitting", self)
        fit_menu.setEnabled(True)

        self.fit_NNLS = QtGui.QAction("NNLS", self)
        self.fit_NNLS.triggered.connect(lambda x: _fit(self, "NNLS"))

        fit_menu.addAction(self.fit_NNLS)

        mono_menu = QtWidgets.QMenu("Mono Exponential", self)
        self.fit_mono = QtGui.QAction("Monoexponential", self)
        self.fit_mono.triggered.connect(lambda x: _fit(self, "mono"))
        mono_menu.addAction(self.fit_mono)

        self.fit_mono_t1 = QtGui.QAction("Monoexponential with T1", self)
        self.fit_mono_t1.triggered.connect(lambda x: _fit(self, "mono_t1"))
        mono_menu.addAction(self.fit_mono_t1)
        # monoMenu.setEnabled(False)
        fit_menu.addMenu(mono_menu)
        menu_bar.addMenu(fit_menu)

        self.fit_multiexp = QtGui.QAction("Multiexponential", self)
        self.fit_multiexp.triggered.connect(lambda x: _fit(self, "multiexp"))
        fit_menu.addAction(self.fit_multiexp)

        # ----- View Menu
        view_menu = QtWidgets.QMenu("&View", self)
        image_menu = QtWidgets.QMenu("Switch Image", self)

        def _switch_image(self, type: str = "Img"):
            """Switch Image Callback"""
            self.settings.setValue("img_disp_type", type)
            self.setup_image()

        self.plt_showImg = QtGui.QAction("Image", self)
        self.plt_showImg.triggered.connect(lambda x: _switch_image(self, "Img"))
        # image_menu.addAction(self.plt_showImg)

        self.plt_showMask = QtGui.QAction("Mask", self)
        self.plt_showMask.triggered.connect(lambda x: _switch_image(self, "Mask"))
        # image_menu.addAction(self.plt_showMask)

        def _plt_show_masked_image(self):
            if self.plt_showMaskedImage.isChecked():
                self.img_overlay.setChecked(False)
                self.img_overlay.setEnabled(False)
                self.settings.setValue("img_disp_overlay", False)
                self.setup_image()
            else:
                self.img_overlay.setEnabled(True)
                self.settings.setValue("img_disp_overlay", True)
                self.setup_image()

        self.plt_showMaskedImage = QtGui.QAction("Image with applied Mask")
        self.plt_showMaskedImage.setEnabled(False)
        self.plt_showMaskedImage.setCheckable(True)
        self.plt_showMaskedImage.setChecked(False)
        self.plt_showMaskedImage.toggled.connect(lambda x: _plt_show_masked_image(self))
        image_menu.addAction(self.plt_showMaskedImage)

        self.plt_showDyn = QtGui.QAction("Dynamic", self)
        self.plt_showDyn.triggered.connect(lambda x: self._switchImage(self, "Dyn"))
        # image_menu.addAction(self.plt_showDyn)
        view_menu.addMenu(image_menu)

        def _plt_show(self):
            """Plot Axis show Callback"""
            if not self.plt_show.isChecked():
                # self.plt_spectrum_canvas.setParent(None)
                # self.plt_spectrum_fig.set_visible(False)
                self.main_hLayout.removeItem(self.plt_vLayout)
                self.settings.setValue("plt_show", True)
            else:
                # self.main_hLayout.addWidget(self.plt_spectrum_canvas)
                self.main_hLayout.addLayout(self.plt_vLayout)
                self.settings.setValue("plt_show", True)
            # self.resizeMainWindow()
            self._resize_figure_axis()
            self._resize_canvas_size()

        self.plt_show = QtGui.QAction("Show Plot")
        self.plt_show.setEnabled(True)
        self.plt_show.setCheckable(True)
        self.plt_show.triggered.connect(lambda x: _plt_show(self))
        view_menu.addAction(self.plt_show)
        view_menu.addSeparator()

        self.plt_DispType_SingleVoxel = QtGui.QAction(
            "Show Single Voxel Spectrum", self
        )
        self.plt_DispType_SingleVoxel.setCheckable(True)
        self.plt_DispType_SingleVoxel.setChecked(True)
        self.settings.setValue("plt_disp_type", "single_voxel")
        self.plt_DispType_SingleVoxel.toggled.connect(
            lambda x: self._switchPlt(self, "single_voxel")
        )
        view_menu.addAction(self.plt_DispType_SingleVoxel)

        self.plt_DispType_SegSpectrum = QtGui.QAction(
            "Show Segmentation Spectrum", self
        )
        self.plt_DispType_SegSpectrum.setCheckable(True)
        self.plt_DispType_SegSpectrum.toggled.connect(
            lambda x: self._switchPlt(self, "seg_spectrum")
        )
        view_menu.addAction(self.plt_DispType_SegSpectrum)
        view_menu.addSeparator()

        def _img_overlay(self):
            """Overlay Callback"""
            self.settings.setValue(
                "img_disp_overlay", True if self.img_overlay.isChecked() else False
            )
            self.setup_image()

        self.img_overlay = QtGui.QAction("Show Mask Overlay", self)
        self.img_overlay.setEnabled(False)
        self.img_overlay.setCheckable(True)
        self.img_overlay.setChecked(False)
        self.settings.setValue("img_disp_overlay", True)
        self.img_overlay.toggled.connect(lambda x: _img_overlay(self))
        view_menu.addAction(self.img_overlay)
        menu_bar.addMenu(view_menu)

        evalMenu = QtWidgets.QMenu("Evaluation", self)
        evalMenu.setEnabled(False)
        # menu_bar.addMenu(evalMenu)

    def _createContextMenu(self):
        self.contextMenu = QtWidgets.QMenu(self)
        plt_menu = QtWidgets.QMenu("Plotting", self)
        plt_menu.addAction(self.plt_show)
        plt_menu.addSeparator()

        plt_menu.addAction(self.plt_DispType_SingleVoxel)

        plt_menu.addAction(self.plt_DispType_SegSpectrum)

        self.contextMenu.addMenu(plt_menu)

    # Events

    def event_filter(self, event):
        if event.button == 1:
            # left mouse button
            if self.data.nii_img.path:
                if event.xdata and event.ydata:
                    # check if point is on image
                    position = [round(event.xdata), round(event.ydata)]
                    # correct inverted y-axis
                    position[1] = self.data.nii_img.array.shape[1] - position[1]
                    self.statusBar.showMessage("(%d, %d)" % (position[0], position[1]))
                    if self.settings.value("plt_show", type=bool):
                        if (
                            self.settings.value("plt_disp_type", type=str)
                            == "single_voxel"
                        ):
                            plotting.show_pixel_signal(
                                self.plt_signal_AX,
                                self.plt_signal_canvas,
                                self.data,
                                self.data.fit.fit_data.fit_params,
                                position,
                            )
                            if np.any(self.data.nii_dyn.array):
                                plotting.show_pixel_spectrum(
                                    self.plt_spectrum_AX,
                                    self.plt_spectrum_canvas,
                                    self.data,
                                    position,
                                )
                        elif (
                            self.settings.value("plt_disp_type", type=str)
                            == "seg_spectrum"
                        ):
                            plotting.show_seg_spectrum(
                                self.plt_spectrum_AX,
                                self.plt_spectrum_canvas,
                                self.data,
                                0,
                            )

    def contextMenuEvent(self, event):
        self.contextMenu.popup(QtGui.QCursor.pos())

    def resizeEvent(self, event):
        self._resize_canvas_size()
        self._resize_figure_axis()

    def _resize_figure_axis(self, aspect_ratio: tuple | None = (1.0, 1.0)):
        """Resize main image axis to canvas size"""
        box = self.img_ax.get_position()
        if box.width > box.height:
            # fix height
            scaling = aspect_ratio[0] / box.width
            new_height = box.height * scaling
            new_y0 = (1 - new_height) / 2
            self.img_ax.set_position(
                [(1 - aspect_ratio[0]) / 2, new_y0, aspect_ratio[1], new_height]
            )
        elif box.width < box.height:
            # fix width
            scaling = aspect_ratio[1] / box.height
            new_width = box.width * scaling
            new_x0 = (1 - new_width) / 2
            self.img_ax.set_position(
                [new_x0, (1 - aspect_ratio[0]) / 2, new_width, aspect_ratio[1]]
            )

    def _resize_canvas_size(self):
        if self.settings.value("plt_show", type=bool):
            canvas_size = self.img_canvas.size()
            self.img_canvas.setMaximumWidth(round(self.width() * 0.6))
            self.SliceSldr.setMaximumWidth(
                round(self.width() * 0.6) - self.SliceSpnBx.width()
            )
            # Canvas size should not exceed 60% of the main windows size so that the graphs can be displayed properly
        else:
            self.img_canvas.setMaximumWidth(16777215)
            self.SliceSldr.setMaximumWidth(16777215)
        # FIXME After deactivating the Plot the Canvas expands but wont fill the whole window

    def _get_image_by_label(self) -> Nii:
        """Get selected Image from settings"""
        if self.settings.value("img_disp_type") == "Img":
            return self.data.nii_img
        elif self.settings.value("img_disp_type") == "Mask":
            return self.data.nii_seg
        elif self.settings.value("img_disp_type") == "Seg":
            return self.data.nii_seg
        elif self.settings.value("img_disp_type") == "Dyn":
            return self.data.nii_dyn

    def setup_image(self):
        """Setup Image on main Axis"""
        self.data.plt.nslice.number = self.SliceSldr.value()
        nii_img = self._get_image_by_label()
        if nii_img.path:
            img_display = nii_img.to_rgba_array(self.data.plt.nslice.value)
            self.img_ax.clear()
            self.img_ax.imshow(img_display, cmap="gray")
            # Add Patches
            if (
                self.settings.value("img_disp_overlay", type=bool)
                and self.data.nii_seg.path
            ):
                nii_seg = self.data.nii_seg
                colors = ["r", "g", "b", "y"]
                if not nii_seg.polygons:
                    nii_seg.calculate_polygons()
                polygon_patches = nii_seg.polygons[self.data.plt.nslice.value]
                for idx in range(nii_seg.number_segs):
                    if polygon_patches:
                        polygon_patch: patches.Polygon
                        for polygon_patch in polygon_patches:
                            if polygon_patch:
                                polygon_patch.set_edgecolor(colors[idx])
                                polygon_patch.set_alpha(self.data.plt.alpha)
                                polygon_patch.set_linewidth(2)
                                # polygon_patch.set_facecolor(colors[idx])
                                polygon_patch.set_facecolor("none")
                                self.img_ax.add_patch(polygon_patch)

            self.img_ax.axis("off")
            self._resize_canvas_size()
            self._resize_figure_axis()
            self.img_canvas.draw()

    def resize_main_window(self):
        # FIXME: main widget should not be larger then 60% of maximum height in case that the image is maxed out
        # NOTE still needed ????
        self.main_hLayout.update()
        self.main_vLayout.update()

    def _b_values_from_dict(self):
        b_values = self.fit_dlg.fitting_dict.pop("b_values", None).value
        if b_values:
            if type(b_values) == str:
                b_values = np.fromstring(
                    b_values.replace("[", "").replace("]", ""), dtype=int, sep="  "
                )
                if b_values.shape != self.data.fit.fit_data.fit_params.b_values.shape:
                    b_values = np.reshape(
                        b_values, self.data.fit.fit_data.fit_params.b_values.shape
                    )
            elif type(b_values) == list:
                b_values = np.array(b_values)

            return b_values

    def _load_b_values(self):
        path = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open B-Value File",
            directory="",
        )[0]

        if path:
            file = Path(path)
            with open(file, "r") as f:
                # find away to decide which one is right
                # self.b_values = np.array([int(x) for x in f.read().split(" ")])
                b_values = [int(x) for x in f.read().split("\n")]
            return b_values


if __name__ == "__main__":
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()  # QtWidgets.QWidget()
    if main_window.settings.value("theme") == "Dark":
        app.setStyle("Fusion")
    main_window.show()
    sys.exit(app.exec())
