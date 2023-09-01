from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path
import numpy as np

from src.utils import Nii, NiiSeg, Processing
from src.ui.promptdlgs import ReshapeSegDlg
from src.ui.settingsdlg import SettingsDlg
from src.ui.fittingdlg import FittingDlg, FittingWidgets, FittingDictionaries
from src.fit import parameters, model


def create_menu_bar(main_window):  #: MainWindow
    menu_bar = main_window.menuBar()

    # ----- File Menu
    file_menu = QtWidgets.QMenu("&File", main_window)

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

    main_window.load_image = QtGui.QAction(
        text="Open &Image...",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(
                Path(main_window.data.app_path), "resources", "PineappleLogo.png"
            ).__str__()
        ),
        # icon=main_window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
    )
    main_window.load_image.triggered.connect(lambda x: _load_image(main_window))
    file_menu.addAction(main_window.load_image)

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

    main_window.load_segmentation = QtGui.QAction(
        text="Open &Segmentation...",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(
                Path(main_window.data.app_path), "resources", "PineappleLogo_Seg.png"
            ).__str__()
        ),
        # icon=main_window.style().standardIcon(
        #     QtWidgets.QStyle.StandardPixmap.SP_FileIcon
        # ),
    )
    main_window.load_segmentation.triggered.connect(lambda x: _load_seg(main_window))
    file_menu.addAction(main_window.load_segmentation)

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

    main_window.load_dyn = QtGui.QAction(
        text="Open &Dynamic Image...",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(
                Path(main_window.data.app_path), "resources", "PineappleLogo_Dyn.png"
            ).__str__()
        ),
        # icon=main_window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
    )
    main_window.load_dyn.triggered.connect(lambda x: _load_dyn(main_window))
    file_menu.addAction(main_window.load_dyn)

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

    main_window.saveImage = QtGui.QAction(
        text="Save Image...",
        parent=main_window,
        icon=main_window.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
        ),
    )
    main_window.saveImage.triggered.connect(lambda x: _save_image(main_window))
    file_menu.addAction(main_window.saveImage)

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

    main_window.saveFitImage = QtGui.QAction(
        text="Save Fit to NifTi...",
        parent=main_window,
        icon=main_window.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
        ),
    )
    main_window.saveFitImage.setEnabled(False)
    main_window.saveFitImage.triggered.connect(lambda x: _save_fit_image(main_window))
    file_menu.addAction(main_window.saveFitImage)

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

    main_window.saveMaskedImage = QtGui.QAction(
        text="Save Masked Image...",
        parent=main_window,
        icon=main_window.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
        ),
    )
    main_window.saveMaskedImage.setEnabled(False)
    main_window.saveMaskedImage.triggered.connect(
        lambda x: _save_masked_image(main_window)
    )
    file_menu.addAction(main_window.saveMaskedImage)

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

    main_window.open_settings_dlg = QtGui.QAction(
        text="Settings...",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(Path(main_window.data.app_path), "resources", "Settings.ico").__str__()
        ),
    )
    main_window.open_settings_dlg.setEnabled(True)
    main_window.open_settings_dlg.triggered.connect(
        lambda x: _open_settings_dlg(main_window)
    )
    file_menu.addAction(main_window.open_settings_dlg)

    menu_bar.addMenu(file_menu)

    # ----- Edit Menu
    edit_menu = QtWidgets.QMenu("&Edit", main_window)
    mask_menu = QtWidgets.QMenu("&Mask Tools", main_window)

    orientation_menu = QtWidgets.QMenu("&Orientation", main_window)
    main_window.rotMask = QtGui.QAction(
        text="&Rotate Mask clockwise",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(Path(main_window.data.app_path), "resources", "rot90.png").__str__()
        ),
    )
    main_window.rotMask.setEnabled(False)
    orientation_menu.addAction(main_window.rotMask)

    # Flip Mask Up Down
    def _mask_flip_up_down(self):
        # Images are rotated 90 degrees so lr and ud are switched
        self.data.nii_seg.array = np.fliplr(self.data.nii_seg.array)
        self.data.nii_seg.calculate_polygons()
        self.setup_image()

    main_window.maskFlipUpDown = QtGui.QAction(
        text="Flip Mask Up-Down",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(Path(main_window.data.app_path), "resources", "flipUD.png").__str__()
        ),
    )
    main_window.maskFlipUpDown.setEnabled(False)
    main_window.maskFlipUpDown.triggered.connect(
        lambda x: _mask_flip_up_down(main_window)
    )
    orientation_menu.addAction(main_window.maskFlipUpDown)

    file_menu.addSeparator()

    # Flip Left Right
    def _mask_flip_left_right(self):
        # Images are rotated 90 degrees so lr and ud are switched
        self.data.nii_seg.array = np.flipud(self.data.nii_seg.array)
        self.data.nii_seg.calculate_polygons()
        self.setup_image()

    main_window.maskFlipLeftRight = QtGui.QAction(
        text="Flip Mask Left-Right",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(Path(main_window.data.app_path), "resources", "flipLR.png").__str__()
        ),
    )
    main_window.maskFlipLeftRight.setEnabled(False)
    main_window.maskFlipLeftRight.triggered.connect(
        lambda x: _mask_flip_left_right(main_window)
    )
    orientation_menu.addAction(main_window.maskFlipLeftRight)

    # Flip Back Forth
    def _mask_flip_back_forth(self):
        self.data.nii_seg.array = np.flip(self.data.nii_seg.array, axis=2)
        self.data.nii_seg.calculate_polygons()
        self.setup_image()

    main_window.maskFlipBackForth = QtGui.QAction(
        text="Flip Mask Back-Forth",
        parent=main_window,
        icon=QtGui.QIcon(
            Path(Path(main_window.data.app_path), "resources", "flipZ.png").__str__()
        ),
    )
    main_window.maskFlipBackForth.setEnabled(False)
    main_window.maskFlipBackForth.triggered.connect(
        lambda x: _mask_flip_back_forth(main_window)
    )
    orientation_menu.addAction(main_window.maskFlipBackForth)

    mask_menu.addMenu(orientation_menu)

    # Mask to Image
    def _mask2img(self):
        self.data.nii_img_masked = Processing.merge_nii_images(
            self.data.nii_img, self.data.nii_seg
        )
        if self.data.nii_img_masked:
            self.plt_showMaskedImage.setEnabled(True)
            self.saveMaskedImage.setEnabled(True)

    main_window.mask2img = QtGui.QAction(text="&Apply on Image", parent=main_window)
    main_window.mask2img.setEnabled(False)
    main_window.mask2img.triggered.connect(lambda x: _mask2img(main_window))
    mask_menu.addAction(main_window.mask2img)

    padding_menu = QtWidgets.QMenu("Zero-Padding", main_window)
    main_window.pad_image = QtGui.QAction(text="For Image", parent=main_window)
    main_window.pad_image.setEnabled(True)
    main_window.pad_image.triggered.connect(main_window.data.nii_img.zero_padding)
    padding_menu.addAction(main_window.pad_image)
    main_window.pad_seg = QtGui.QAction(text="For Segmentation", parent=main_window)
    main_window.pad_seg.setEnabled(True)
    # self.pad_seg.triggerd.connect(self.data.nii_seg.super().zero_padding)
    # self.pad_seg.triggered.connect(lambda x: _mask2img(self))
    padding_menu.addAction(main_window.pad_seg)
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
    fit_menu = QtWidgets.QMenu("&Fitting", main_window)
    fit_menu.setEnabled(True)

    main_window.fit_NNLS = QtGui.QAction(text="NNLS", parent=main_window)
    main_window.fit_NNLS.triggered.connect(lambda x: _fit(main_window, "NNLS"))
    fit_menu.addAction(main_window.fit_NNLS)

    main_window.fit_mono = QtGui.QAction(text="Mono-exponential", parent=main_window)
    main_window.fit_mono.triggered.connect(lambda x: _fit(main_window, "mono"))
    fit_menu.addAction(main_window.fit_mono)

    main_window.fit_multiExp = QtGui.QAction(
        text="Multi-exponential", parent=main_window
    )
    main_window.fit_multiExp.triggered.connect(lambda x: _fit(main_window, "multiExp"))
    fit_menu.addAction(main_window.fit_multiExp)

    menu_bar.addMenu(fit_menu)

    # ----- View Menu
    view_menu = QtWidgets.QMenu("&View", main_window)
    image_menu = QtWidgets.QMenu("Switch Image", main_window)

    def _switch_image(self, img_type: str = "Img"):
        """Switch Image Callback"""
        self.settings.setValue("img_disp_type", img_type)
        self.setup_image()

    main_window.plt_showImg = QtGui.QAction(text="Image", parent=main_window)
    main_window.plt_showImg.triggered.connect(
        lambda x: _switch_image(main_window, "Img")
    )
    # image_menu.addAction(self.plt_showImg)

    main_window.plt_showMask = QtGui.QAction(text="Mask", parent=main_window)
    main_window.plt_showMask.triggered.connect(
        lambda x: _switch_image(main_window, "Mask")
    )
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

    main_window.plt_showMaskedImage = QtGui.QAction("Image with applied Mask")
    main_window.plt_showMaskedImage.setEnabled(False)
    main_window.plt_showMaskedImage.setCheckable(True)
    main_window.plt_showMaskedImage.setChecked(False)
    main_window.plt_showMaskedImage.toggled.connect(
        lambda x: _plt_show_masked_image(main_window)
    )
    image_menu.addAction(main_window.plt_showMaskedImage)

    # main_window.plt_showDyn = QtGui.QAction(text="Dynamic", parent=main_window)
    # main_window.plt_showDyn.triggered.connect(
    #     lambda x: _switch_image(main_window, "Dyn")
    # )
    # image_menu.addAction(main_window.plt_showDyn)
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

    main_window.plt_show = QtGui.QAction("Show Plot")
    main_window.plt_show.setEnabled(True)
    main_window.plt_show.setCheckable(True)
    main_window.plt_show.triggered.connect(lambda x: _plt_show(main_window))
    view_menu.addAction(main_window.plt_show)
    view_menu.addSeparator()

    main_window.plt_DispType_SingleVoxel = QtGui.QAction(
        text="Show Single Voxel Spectrum", parent=main_window
    )
    main_window.plt_DispType_SingleVoxel.setCheckable(True)
    main_window.plt_DispType_SingleVoxel.setChecked(True)
    main_window.settings.setValue("plt_disp_type", "single_voxel")
    # main_window.plt_DispType_SingleVoxel.toggled.connect(
    #     lambda x: main_window._switchPlt(main_window, "single_voxel")
    # )
    main_window.plt_DispType_SingleVoxel.setEnabled(False)
    view_menu.addAction(main_window.plt_DispType_SingleVoxel)

    main_window.plt_DispType_SegSpectrum = QtGui.QAction(
        text="Show Segmentation Spectrum", parent=main_window
    )
    main_window.plt_DispType_SegSpectrum.setCheckable(True)
    # main_window.plt_DispType_SegSpectrum.toggled.connect(
    #     lambda x: main_window._switchPlt(main_window, "seg_spectrum")
    # )
    main_window.plt_DispType_SegSpectrum.setEnabled(False)
    view_menu.addAction(main_window.plt_DispType_SegSpectrum)
    view_menu.addSeparator()

    def _img_overlay(self):
        """Overlay Callback"""
        self.settings.setValue(
            "img_disp_overlay", True if self.img_overlay.isChecked() else False
        )
        self.setup_image()

    main_window.img_overlay = QtGui.QAction(
        text="Show Mask Overlay", parent=main_window
    )
    main_window.img_overlay.setEnabled(False)
    main_window.img_overlay.setCheckable(True)
    main_window.img_overlay.setChecked(False)
    main_window.settings.setValue("img_disp_overlay", True)
    main_window.img_overlay.toggled.connect(lambda x: _img_overlay(main_window))
    view_menu.addAction(main_window.img_overlay)
    menu_bar.addMenu(view_menu)

    eval_menu = QtWidgets.QMenu("Evaluation", main_window)
    eval_menu.setEnabled(False)
    # menu_bar.addMenu(eval_menu)
