from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path
import numpy as np

from src.utils import Nii, NiiSeg, Processing
from src.ui.promptdlgs import ReshapeSegDlg
from src.ui.settingsdlg import SettingsDlg
from src.ui.fittingdlg import FittingDlg, FittingWidgets, FittingDictionaries
from src.fit import parameters, model


def create_menu_bar(parent):  #: MainWindow
    # ----- Setup Menubar
    menu_bar = parent.menuBar()
    # ----- File Menu
    file_menu = QtWidgets.QMenu("&File", parent)

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
                self.img_overlay.setEnabled(True if self.data.nii_seg.path else False)
            else:
                print("Warning no file selected")

    # img = Path(Path(parent.data.app_path), "resources", "PineappleLogo.png").__str__()
    parent.load_image = QtGui.QAction(
        parent.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
        "Open &Image...",
        parent,
    )
    parent.load_image.triggered.connect(lambda x: _load_image(parent))
    file_menu.addAction(parent.load_image)

    # Add Load Image Icon
    img = Path(Path(parent.data.app_path), "resources", "PineappleLogo.png").__str__()
    parent.load_image.setIcon(QtGui.QIcon(img))

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

                self.img_overlay.setEnabled(True if self.data.nii_seg.path else False)
                self.img_overlay.setChecked(True if self.data.nii_seg.path else False)
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

    parent.load_segmentation = QtGui.QAction(
        parent.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
        "Open &Segmentation...",
        parent,
    )
    parent.load_segmentation.triggered.connect(lambda x: _load_seg(parent))
    file_menu.addAction(parent.load_segmentation)

    # Add Segmentation Image Icon
    img = Path(
        Path(parent.data.app_path), "resources", "PineappleLogo_Seg.png"
    ).__str__()
    parent.load_segmentation.setIcon(QtGui.QIcon(img))

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

    parent.load_dyn = QtGui.QAction(
        parent.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
        "Open &Dynamic Image...",
        parent,
    )
    parent.load_dyn.triggered.connect(lambda x: _load_dyn(parent))
    file_menu.addAction(parent.load_dyn)

    # Add Load Dynamic Image Icon
    img = Path(
        Path(parent.data.app_path), "resources", "PineappleLogo_Dyn.png"
    ).__str__()
    parent.load_dyn.setIcon(QtGui.QIcon(img))

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

    parent.saveImage = QtGui.QAction(
        parent.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
        ),
        "Save Image...",
        parent,
    )
    parent.saveImage.triggered.connect(lambda x: _save_image(parent))
    file_menu.addAction(parent.saveImage)

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

    parent.saveFitImage = QtGui.QAction(
        parent.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
        ),
        "Save Fit to NifTi...",
        parent,
    )
    parent.saveFitImage.setEnabled(False)
    parent.saveFitImage.triggered.connect(lambda x: _save_fit_image(parent))
    file_menu.addAction(parent.saveFitImage)

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

    parent.saveMaskedImage = QtGui.QAction(
        parent.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
        ),
        "Save Masked Image...",
        parent,
    )
    parent.saveMaskedImage.setEnabled(False)
    parent.saveMaskedImage.triggered.connect(lambda x: _save_masked_image(parent))
    file_menu.addAction(parent.saveMaskedImage)

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

    parent.open_settings_dlg = QtGui.QAction(
        parent.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_TitleBarMenuButton
        ),
        "Settings...",
        parent,
    )
    parent.open_settings_dlg.setEnabled(True)
    parent.open_settings_dlg.triggered.connect(lambda x: _open_settings_dlg(parent))
    file_menu.addAction(parent.open_settings_dlg)

    menu_bar.addMenu(file_menu)

    # ----- Edit Menu
    edit_menu = QtWidgets.QMenu("&Edit", parent)
    mask_menu = QtWidgets.QMenu("&Mask Tools", parent)

    orientation_menu = QtWidgets.QMenu("&Orientation", parent)
    parent.rotMask = QtGui.QAction("&Rotate Mask clockwise", parent)
    parent.rotMask.setEnabled(False)
    orientation_menu.addAction(parent.rotMask)
    # Add Icon
    img = Path(Path(parent.data.app_path), "resources", "rot90.png").__str__()
    parent.rotMask.setIcon(QtGui.QIcon(img))

    # Flip Mask Up Down
    def _mask_flip_up_down(self):
        # Images are rotated 90 degrees so lr and ud are switched
        self.data.nii_seg.array = np.fliplr(self.data.nii_seg.array)
        self.data.nii_seg.calculate_polygons()
        self.setup_image()

    parent.maskFlipUpDown = QtGui.QAction("Flip Mask Up-Down", parent)
    parent.maskFlipUpDown.setEnabled(False)
    parent.maskFlipUpDown.triggered.connect(lambda x: _mask_flip_up_down(parent))
    orientation_menu.addAction(parent.maskFlipUpDown)
    img = Path(Path(parent.data.app_path), "resources", "flipUD.png").__str__()
    parent.maskFlipUpDown.setIcon(QtGui.QIcon(img))

    file_menu.addSeparator()

    # Flip Left Right
    def _mask_flip_left_right(self):
        # Images are rotated 90 degrees so lr and ud are switched
        self.data.nii_seg.array = np.flipud(self.data.nii_seg.array)
        self.data.nii_seg.calculate_polygons()
        self.setup_image()

    parent.maskFlipLeftRight = QtGui.QAction("Flip Mask Left-Right", parent)
    parent.maskFlipLeftRight.setEnabled(False)
    parent.maskFlipLeftRight.triggered.connect(lambda x: _mask_flip_left_right(parent))
    orientation_menu.addAction(parent.maskFlipLeftRight)
    img = Path(Path(parent.data.app_path), "resources", "flipLR.png").__str__()
    parent.maskFlipLeftRight.setIcon(QtGui.QIcon(img))

    # Flip Back Forth
    def _mask_flip_back_forth(self):
        self.data.nii_seg.array = np.flip(self.data.nii_seg.array, axis=2)
        self.data.nii_seg.calculate_polygons()
        self.setup_image()

    parent.maskFlipBackForth = QtGui.QAction("Flip Mask Back-Forth", parent)
    parent.maskFlipBackForth.setEnabled(False)
    parent.maskFlipBackForth.triggered.connect(lambda x: _mask_flip_back_forth(parent))
    orientation_menu.addAction(parent.maskFlipBackForth)
    img = Path(Path(parent.data.app_path), "resources", "flipZ.png").__str__()
    parent.maskFlipBackForth.setIcon(QtGui.QIcon(img))

    mask_menu.addMenu(orientation_menu)

    # Mask to Image
    def _mask2img(self):
        self.data.nii_img_masked = Processing.merge_nii_images(
            self.data.nii_img, self.data.nii_seg
        )
        if self.data.nii_img_masked:
            self.plt_showMaskedImage.setEnabled(True)
            self.saveMaskedImage.setEnabled(True)

    parent.mask2img = QtGui.QAction("&Apply on Image", parent)
    parent.mask2img.setEnabled(False)
    parent.mask2img.triggered.connect(lambda x: _mask2img(parent))
    mask_menu.addAction(parent.mask2img)

    padding_menu = QtWidgets.QMenu("Zero-Padding", parent)
    parent.pad_image = QtGui.QAction("For Image", parent)
    parent.pad_image.setEnabled(True)
    parent.pad_image.triggered.connect(parent.data.nii_img.zero_padding)
    padding_menu.addAction(parent.pad_image)
    parent.pad_seg = QtGui.QAction("For Segmentation", parent)
    parent.pad_seg.setEnabled(True)
    # self.pad_seg.triggerd.connect(self.data.nii_seg.super().zero_padding)
    # self.pad_seg.triggered.connect(lambda x: _mask2img(self))
    padding_menu.addAction(parent.pad_seg)
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
                fit_data.fit_pixel_wise(
                    multi_threading=self.settings.value("multithreading", type=bool)
                )
                self.data.nii_dyn = Nii().from_array(
                    self.data.fit_data.fit_results.spectrum
                )

            elif fit_data.fit_area == "Segmentation":
                fit_data.fit_segmentation_wise()

            self.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

            self.saveFitImage.setEnabled(True)

    # ----- Fitting Menu
    fit_menu = QtWidgets.QMenu("&Fitting", parent)
    fit_menu.setEnabled(True)

    parent.fit_NNLS = QtGui.QAction("NNLS", parent)
    parent.fit_NNLS.triggered.connect(lambda x: _fit(parent, "NNLS"))
    fit_menu.addAction(parent.fit_NNLS)

    parent.fit_mono = QtGui.QAction("Mono-exponential", parent)
    parent.fit_mono.triggered.connect(lambda x: _fit(parent, "mono"))
    fit_menu.addAction(parent.fit_mono)

    parent.fit_multiExp = QtGui.QAction("Multi-exponential", parent)
    parent.fit_multiExp.triggered.connect(lambda x: _fit(parent, "multiExp"))
    fit_menu.addAction(parent.fit_multiExp)

    menu_bar.addMenu(fit_menu)

    # ----- View Menu
    view_menu = QtWidgets.QMenu("&View", parent)
    image_menu = QtWidgets.QMenu("Switch Image", parent)

    def _switch_image(self, type: str = "Img"):
        """Switch Image Callback"""
        self.settings.setValue("img_disp_type", type)
        self.setup_image()

    parent.plt_showImg = QtGui.QAction("Image", parent)
    parent.plt_showImg.triggered.connect(lambda x: _switch_image(parent, "Img"))
    # image_menu.addAction(self.plt_showImg)

    parent.plt_showMask = QtGui.QAction("Mask", parent)
    parent.plt_showMask.triggered.connect(lambda x: _switch_image(parent, "Mask"))
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

    parent.plt_showMaskedImage = QtGui.QAction("Image with applied Mask")
    parent.plt_showMaskedImage.setEnabled(False)
    parent.plt_showMaskedImage.setCheckable(True)
    parent.plt_showMaskedImage.setChecked(False)
    parent.plt_showMaskedImage.toggled.connect(lambda x: _plt_show_masked_image(parent))
    image_menu.addAction(parent.plt_showMaskedImage)

    parent.plt_showDyn = QtGui.QAction("Dynamic", parent)
    parent.plt_showDyn.triggered.connect(lambda x: parent._switchImage(parent, "Dyn"))
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

    parent.plt_show = QtGui.QAction("Show Plot")
    parent.plt_show.setEnabled(True)
    parent.plt_show.setCheckable(True)
    parent.plt_show.triggered.connect(lambda x: _plt_show(parent))
    view_menu.addAction(parent.plt_show)
    view_menu.addSeparator()

    parent.plt_DispType_SingleVoxel = QtGui.QAction(
        "Show Single Voxel Spectrum", parent
    )
    parent.plt_DispType_SingleVoxel.setCheckable(True)
    parent.plt_DispType_SingleVoxel.setChecked(True)
    parent.settings.setValue("plt_disp_type", "single_voxel")
    parent.plt_DispType_SingleVoxel.toggled.connect(
        lambda x: parent._switchPlt(parent, "single_voxel")
    )
    view_menu.addAction(parent.plt_DispType_SingleVoxel)

    parent.plt_DispType_SegSpectrum = QtGui.QAction(
        "Show Segmentation Spectrum", parent
    )
    parent.plt_DispType_SegSpectrum.setCheckable(True)
    parent.plt_DispType_SegSpectrum.toggled.connect(
        lambda x: parent._switchPlt(parent, "seg_spectrum")
    )
    view_menu.addAction(parent.plt_DispType_SegSpectrum)
    view_menu.addSeparator()

    def _img_overlay(self):
        """Overlay Callback"""
        self.settings.setValue(
            "img_disp_overlay", True if self.img_overlay.isChecked() else False
        )
        self.setup_image()

    parent.img_overlay = QtGui.QAction("Show Mask Overlay", parent)
    parent.img_overlay.setEnabled(False)
    parent.img_overlay.setCheckable(True)
    parent.img_overlay.setChecked(False)
    parent.settings.setValue("img_disp_overlay", True)
    parent.img_overlay.toggled.connect(lambda x: _img_overlay(parent))
    view_menu.addAction(parent.img_overlay)
    menu_bar.addMenu(view_menu)

    eval_menu = QtWidgets.QMenu("Evaluation", parent)
    eval_menu.setEnabled(False)
    # menu_bar.addMenu(eval_menu)
