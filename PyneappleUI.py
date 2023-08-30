import os
import sys

from multiprocessing import freeze_support

from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
from PIL import Image
import numpy as np

import src.plotting as plotting
from src.fit import parameters, model
from src.ui.fittingdlg import FittingDlg, FittingWidgets, FittingDictionaries
from src.ui.promptdlgs import ReshapeSegDlg
from src.ui.settingsdlg import SettingsDlg
from src.utils import Nii, NiiSeg
from src.appdata import AppData


# v0.4.3


# noinspection PyUnresolvedReferences
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: Path | str = None) -> None:
        super(MainWindow, self).__init__()
        self.data = AppData()
        # self.data.fit.fit_data = fit.FitData()
        # Load Settings
        self._load_settings()

        # initiate UI
        self._setup_ui()

        self.fit_dlg = FittingDlg
        self.settings_dlg = SettingsDlg
        # if path:
        #     self._load_image(path)

    def _load_settings(self):
        self.settings = QtCore.QSettings("MyApp", "Pyneapple")
        if self.settings.value("last_dir", "") == "":
            self.settings.setValue("last_dir", os.path.abspath(__file__))
            self.settings.setValue("theme", "Light")  # "Dark", "Light"

            self.settings.setValue("default_seg_alpha", 0.5)
        self.settings.setValue("plt_show", False)

        if not self.settings.value("default_seg_colors", type=list):
            self.settings.setValue(
                "default_seg_colors", ["#ff0000", "#0000ff", "#00ff00", "#ffff00"]
            )
        self.data.plt["seg_colors"] = self.settings.value(
            "default_seg_colors", type=list
        )

        if not self.settings.value("default_seg_alpha", type=float):
            self.settings.setValue("default_seg_alpha", 0.5)
        self.data.plt["seg_alpha"] = self.settings.value(
            "default_seg_alpha", type=float
        )

        if not self.settings.value("default_seg_line_width", type=float):
            self.settings.setValue("default_seg_line_width", 2.0)
        self.data.plt["seg_line_width"] = self.settings.value(
            "default_seg_line_width", type=float
        )

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
        self._create_context_menu()

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
        theme = self.settings.value("theme")
        if theme == "Dark" or theme == "Fusion":
            # QtWidgets.QApplication.setStyle("Fusion")
            self.img_ax.imshow(
                Image.open(
                    Path(
                        Path(__file__).parent,
                        "resources",
                        "PyNeappleLogo_gray_text.png",
                    )
                ),
                cmap="gray",
            )
            self.img_fig.set_facecolor("black")
        elif theme == "Light":
            # QtWidgets.QApplication.setStyle("Windows")
            self.img_ax.imshow(
                Image.open(
                    Path(
                        Path(__file__).parent,
                        "resources",
                        "PyNeappleLogo_gray_text.png",
                    )
                ),
                cmap="gray",
            )
        self.img_ax.axis("off")

        # ----- Slider
        def _slice_slider_changed(self):
            """Slice Slider Callback"""
            self.data.plt["n_slice"].number = self.SliceSlider.value()
            self.SliceSpnBx.setValue(self.SliceSlider.value())
            self.setup_image()

        self.SliceHLayout = QtWidgets.QHBoxLayout()  # Layout for Slider ans Spinbox
        self.SliceSlider = QtWidgets.QSlider()
        self.SliceSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.SliceSlider.setEnabled(False)
        self.SliceSlider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.SliceSlider.setTickInterval(1)
        self.SliceSlider.setMinimum(1)
        self.SliceSlider.setMaximum(20)
        self.SliceSlider.valueChanged.connect(lambda x: _slice_slider_changed(self))
        self.SliceHLayout.addWidget(self.SliceSlider)

        # ----- SpinBox
        def _slice_spn_bx_changed(self):
            """Slice Spinbox Callback"""
            self.data.plt["n_slice"].number = self.SliceSpnBx.value()
            self.SliceSlider.setValue(self.SliceSpnBx.value())
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
                    directory="data",
                    filter="NifTi (*.nii *.nii.gz)",
                )[0]
            if path:
                file = Path(path) if path else None
                self.data.nii_img = Nii(file)
                if self.data.nii_img.path is not None:
                    self.data.plt["n_slice"].number = self.SliceSlider.value()
                    self.SliceSlider.setEnabled(True)
                    self.SliceSlider.setMaximum(self.data.nii_img.array.shape[2])
                    self.SliceSpnBx.setEnabled(True)
                    self.SliceSpnBx.setMaximum(self.data.nii_img.array.shape[2])
                    self.settings.setValue("img_disp_type", "Img")
                    self.setup_image()
                    self.mask2img.setEnabled(True if self.data.nii_seg.path else False)
                    self.img_overlay.setEnabled(
                        True if self.data.nii_seg.path else False
                    )
                else:
                    print("Warning no file selected")

        self.load_image = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Image...",
            self,
        )
        self.load_image.triggered.connect(lambda x: _load_image(self))
        file_menu.addAction(self.load_image)

        # Load Segmentation
        def _load_seg(self):
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
                    if self.data.nii_img.path:
                        if (
                            not self.data.nii_img.array.shape[:3]
                            == self.data.nii_seg.array.shape[:3]
                        ):
                            print("Warning: Image and segmentation shape do not match!")
                            reshape_seg_dlg = ReshapeSegDlg(
                                self.data.nii_img, self.data.nii_seg
                            )
                            result = reshape_seg_dlg.exec()
                            if result == QtWidgets.QDialog.accepted or result:
                                self.data.nii_seg = reshape_seg_dlg.new_seg
                                self.setup_image()
                            else:
                                print(
                                    "Warning: Img and segmentation shape missmatch still present!"
                                )
            else:
                print("Warning: No file selected")

        self.load_segmentation = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Segmentation...",
            self,
        )
        self.load_segmentation.triggered.connect(lambda x: _load_seg(self))
        file_menu.addAction(self.load_segmentation)

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

        self.load_dyn = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "Open &Dynamic Image...",
            self,
        )
        self.load_dyn.triggered.connect(lambda x: _load_dyn(self))
        file_menu.addAction(self.load_dyn)
        file_menu.addSeparator()

        # Save Image
        def _save_image(self):
            file_name = self.data.nii_img.path
            file = Path(
                QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save Image",
                    file_name.__str__(),
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
            self.settings_dlg = SettingsDlg(
                self.settings,
                self.data.plt
                # SettingsDictionary.get_settings_dict(self.data)
            )
            self.settings_dlg.exec()
            self.settings, self.data = self.settings_dlg.get_settings_data(self.data)
            self.change_theme()
            self.setup_image()

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

        padding_menu = QtWidgets.QMenu("Zero-Padding", self)
        self.pad_image = QtGui.QAction("For Image", self)
        self.pad_image.setEnabled(True)
        self.pad_image.triggered.connect(self.data.nii_img.zero_padding)
        padding_menu.addAction(self.pad_image)
        self.pad_seg = QtGui.QAction("For Segmentation", self)
        self.pad_seg.setEnabled(True)
        # self.pad_seg.triggerd.connect(self.data.nii_seg.super().zero_padding)
        # self.pad_seg.triggered.connect(lambda x: _mask2img(self))
        padding_menu.addAction(self.pad_seg)
        edit_menu.addMenu(padding_menu)

        edit_menu.addMenu(mask_menu)

        menu_bar.addMenu(edit_menu)

        # ----- Fitting Procedure
        def _fit(self, model_name: str):
            fit_data = self.data.fit_data
            dlg_dict = dict()
            if model_name == ("NNLS" or "NNLSreg"):
                if not (type(fit_data.fit_params) == parameters.NNLSregParams):
                    fit_data.fit_params = parameters.NNLSregParams()
                fit_data.model_name = "NNLS"
                # Prepare Dlg Dict
                dlg_dict = FittingDictionaries.get_nnls_dict(fit_data.fit_params)

            if model_name == ("mono" or "mono_t1"):
                # BUG @JJ this should look like the following but it wont work and i dont get it
                # if not (
                #     type(fit_data.fit_params) == parameters.MonoParams
                #     or type(fit_data.fit_params) == parameters.MonoT1Params
                # ):
                #     fit_data.fit_params = parameters.MonoParams()
                if (
                    type(fit_data.fit_params) == parameters.Parameters
                    or type(fit_data.fit_params) == parameters.NNLSregParams
                ):
                    fit_data.fit_params = parameters.MonoParams()
                dlg_dict = FittingDictionaries.get_mono_dict(fit_data.fit_params)
                fit_data.model_name = "mono"

            if model_name == "multiExp":
                # fit_data = self.data.multiExp
                fit_data.fit_params = parameters.MultiExpParams()
                fit_data.model_name = "multiExp"
                dlg_dict = FittingDictionaries.get_multi_exp_dict(fit_data.fit_params)

            dlg_dict["b_values"] = FittingWidgets.PushButton(
                "Load B-Values",
                str(fit_data.fit_params.b_values),
                button_function=self._load_b_values,
                button_text="Open File",
            )

            # Launch Dlg
            self.fit_dlg = FittingDlg(model_name, dlg_dict)
            self.fit_dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self.fit_dlg.exec()

            # Check for T1 advanced fitting if TM is set
            if model_name == "mono" and dlg_dict["TM"].value:
                fit_data.fit_params = parameters.MonoT1Params()

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
                        self.data.fit_data.fit_results.spectrum
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

        self.fit_mono = QtGui.QAction("Mono-exponential", self)
        self.fit_mono.triggered.connect(lambda x: _fit(self, "mono"))
        fit_menu.addAction(self.fit_mono)

        self.fit_multiExp = QtGui.QAction("Multi-exponential", self)
        self.fit_multiExp.triggered.connect(lambda x: _fit(self, "multiExp"))
        fit_menu.addAction(self.fit_multiExp)

        menu_bar.addMenu(fit_menu)

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
            self.setup_image()

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

        eval_menu = QtWidgets.QMenu("Evaluation", self)
        eval_menu.setEnabled(False)
        # menu_bar.addMenu(eval_menu)

    def _save_slice(self):
        if self.data.nii_img.path:
            file_name = self.data.nii_img.path
            new_name = file_name.parent / (file_name.stem + ".png")

            file_path = Path(
                QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save slice image:",
                    new_name.__str__(),
                    "PNG Files (*.png)",
                )[0]
            )
        else:
            file_path = None

        if file_path:
            self.img_fig.savefig(file_path, bbox_inches="tight", pad_inches=0)
            print("Figure saved:", file_path)

    def _create_context_menu(self):
        self.context_menu = QtWidgets.QMenu(self)
        plt_menu = QtWidgets.QMenu("Plotting", self)
        plt_menu.addAction(self.plt_show)
        plt_menu.addSeparator()
        plt_menu.addAction(self.plt_DispType_SingleVoxel)
        plt_menu.addAction(self.plt_DispType_SegSpectrum)

        self.context_menu.addMenu(plt_menu)
        self.context_menu.addSeparator()

        self.save_slice = QtGui.QAction("Save slice to image", self)
        self.save_slice.triggered.connect(self._save_slice)
        self.context_menu.addAction(self.save_slice)

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
                                self.data.fit_data.fit_params,
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
        self.context_menu.popup(QtGui.QCursor.pos())

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
            self.SliceSlider.setMaximumWidth(
                round(self.width() * 0.6) - self.SliceSpnBx.width()
            )
            # Canvas size should not exceed 60% of the main windows size so that the graphs can be displayed properly
        else:
            self.img_canvas.setMaximumWidth(16777215)
            self.SliceSlider.setMaximumWidth(16777215)
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
        self.data.plt["n_slice"].number = self.SliceSlider.value()
        nii_img = self._get_image_by_label()
        if nii_img.path:
            img_display = nii_img.to_rgba_array(self.data.plt["n_slice"].value)
            self.img_ax.clear()
            self.img_ax.imshow(img_display, cmap="gray")
            # Add Patches
            if (
                self.settings.value("img_disp_overlay", type=bool)
                and self.data.nii_seg.path
            ):
                nii_seg = self.data.nii_seg
                colors = self.data.plt["seg_colors"]
                if nii_seg.segmentations:
                    seg_color_idx = 0
                    for seg_number in nii_seg.segmentations:
                        segmentation = nii_seg.segmentations[seg_number]
                        if segmentation.polygon_patches[self.data.plt["n_slice"].value]:
                            polygon_patch: patches.Polygon
                            for polygon_patch in segmentation.polygon_patches[
                                self.data.plt["n_slice"].value
                            ]:
                                if not colors[seg_color_idx] == "None":
                                    polygon_patch.set_edgecolor(colors[seg_color_idx])
                                else:
                                    polygon_patch.set_edgecolor("none")
                                polygon_patch.set_alpha(self.data.plt["seg_alpha"])
                                polygon_patch.set_linewidth(
                                    self.data.plt["seg_line_width"]
                                )
                                # polygon_patch.set_facecolor(colors[seg_color_idx])
                                polygon_patch.set_facecolor("none")
                                self.img_ax.add_patch(polygon_patch)
                        seg_color_idx += 1

                #     nii_seg.calculate_polygons()
                # polygon_patches = nii_seg.segmentations[self.data.plt.n_slice.value]
                # for idx in range(nii_seg.n_segmentations):
                #     if polygon_patches:
                #         polygon_patch: patches.Polygon
                #         for polygon_patch in polygon_patches:
                #             if polygon_patch:
                #                 polygon_patch.set_edgecolor(colors[idx])
                #                 polygon_patch.set_alpha(self.data.plt.alpha)
                #                 polygon_patch.set_linewidth(2)
                #                 # polygon_patch.set_facecolor(colors[idx])
                #                 polygon_patch.set_facecolor("none")
                #                 self.img_ax.add_patch(polygon_patch)

            self.img_ax.axis("off")
            self._resize_canvas_size()
            self._resize_figure_axis()
            self.img_canvas.draw()

    def _b_values_from_dict(self):
        b_values = self.fit_dlg.fitting_dict.pop("b_values", None).value
        if b_values:
            if type(b_values) == str:
                b_values = np.fromstring(
                    b_values.replace("[", "").replace("]", ""), dtype=int, sep="  "
                )
                if b_values.shape != self.data.fit_data.fit_params.b_values.shape:
                    b_values = np.reshape(
                        b_values, self.data.fit_data.fit_params.b_values.shape
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

    def change_theme(self):
        if self.settings.value("theme") == "Dark":
            QtWidgets.QApplication.setStyle("Fusion")
            if not self.data.nii_img.path:
                self.img_ax.imshow(
                    Image.open(
                        # Path(Path(__file__).parent, "resources", "noImage_white.png")
                        Path(
                            Path(__file__).parent,
                            "resources",
                            "PyNeappleLogo_gray.png",
                        )
                    ),
                    cmap="gray",
                )
                self.img_fig.set_facecolor("black")
        elif self.settings.value("theme") == "Light":
            QtWidgets.QApplication.setStyle("windowsvista")


if __name__ == "__main__":
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()  # QtWidgets.QWidget()
    main_window.change_theme()
    main_window.show()
    sys.exit(app.exec())
