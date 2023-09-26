from __future__ import annotations
from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path
import numpy as np

from src.utils import Nii, NiiSeg, Processing
from src.ui.promptdlgs import ReshapeSegDlg, FitParametersDlg
from src.ui.settingsdlg import SettingsDlg
from src.ui.fittingdlg import FittingDlg, FittingDictionaries, FittingWidgets
from src.fit import parameters, model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyneappleUI import MainWindow


class MenuBar(object):
    @staticmethod
    def setup_menubar(parent):
        # ----- File Menu
        file_menu = QtWidgets.QMenu("&File", parent)

        # Load Image
        parent.load_image = QtGui.QAction(
            text="Open &Image...",
            parent=parent,
            icon=QtGui.QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "PineappleLogo.png"
                ).__str__()
            ),
            # icon=main_window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
        )
        parent.load_image.triggered.connect(lambda x: MenuBar._load_image(parent))
        file_menu.addAction(parent.load_image)

        # Load Segmentation
        parent.load_segmentation = QtGui.QAction(
            text="Open &Segmentation...",
            parent=parent,
            icon=QtGui.QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "PineappleLogo_Seg.png",
                ).__str__()
            ),
            # icon=main_window.style().standardIcon(
            #     QtWidgets.QStyle.StandardPixmap.SP_FileIcon
            # ),
        )
        parent.load_segmentation.triggered.connect(lambda x: MenuBar._load_seg(parent))
        file_menu.addAction(parent.load_segmentation)

        # Load dynamic Image
        parent.load_dyn = QtGui.QAction(
            text="Open &Dynamic Image...",
            parent=parent,
            icon=QtGui.QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "PineappleLogo_Dyn.png",
                ).__str__()
            ),
            # icon=main_window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
        )
        parent.load_dyn.triggered.connect(lambda x: MenuBar._load_dyn(parent))
        file_menu.addAction(parent.load_dyn)

        file_menu.addSeparator()

        # Save Image
        parent.saveImage = QtGui.QAction(
            text="Save Image...",
            parent=parent,
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        parent.saveImage.triggered.connect(lambda x: MenuBar._save_image(parent))
        file_menu.addAction(parent.saveImage)

        # Save Fit Image
        parent.saveFitImage = QtGui.QAction(
            text="Save Fit to NifTi...",
            parent=parent,
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        parent.saveFitImage.setEnabled(False)
        parent.saveFitImage.triggered.connect(lambda x: MenuBar._save_fit_image(parent))
        file_menu.addAction(parent.saveFitImage)

        # Save masked image
        parent.saveMaskedImage = QtGui.QAction(
            text="Save Masked Image...",
            parent=parent,
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        parent.saveMaskedImage.setEnabled(False)
        parent.saveMaskedImage.triggered.connect(
            lambda x: MenuBar._save_masked_image(parent)
        )
        file_menu.addAction(parent.saveMaskedImage)

        file_menu.addSeparator()

        # Open Settings
        parent.open_settings_dlg = QtGui.QAction(
            text="Settings...",
            parent=parent,
            icon=QtGui.QIcon(
                Path(Path(parent.data.app_path), "resources", "Settings.ico").__str__()
            ),
        )
        parent.open_settings_dlg.setEnabled(True)
        parent.open_settings_dlg.triggered.connect(
            lambda x: MenuBar._open_settings_dlg(parent)
        )
        file_menu.addAction(parent.open_settings_dlg)

        parent.menuBar().addMenu(file_menu)

        # ----- Edit Menu
        edit_menu = QtWidgets.QMenu("&Edit", parent)
        mask_menu = QtWidgets.QMenu("&Mask Tools", parent)

        orientation_menu = QtWidgets.QMenu("&Orientation", parent)
        parent.rotMask = QtGui.QAction(
            text="&Rotate Mask clockwise",
            parent=parent,
            icon=QtGui.QIcon(
                Path(Path(parent.data.app_path), "resources", "rot90.png").__str__()
            ),
        )
        parent.rotMask.setEnabled(False)
        orientation_menu.addAction(parent.rotMask)

        # Flip Mask Up Down
        parent.maskFlipUpDown = QtGui.QAction(
            text="Flip Mask Up-Down",
            parent=parent,
            icon=QtGui.QIcon(
                Path(Path(parent.data.app_path), "resources", "flipUD.png").__str__()
            ),
        )
        parent.maskFlipUpDown.setEnabled(False)
        parent.maskFlipUpDown.triggered.connect(
            lambda x: MenuBar._mask_flip_up_down(parent)
        )
        orientation_menu.addAction(parent.maskFlipUpDown)

        file_menu.addSeparator()

        # Flip Left Right
        parent.maskFlipLeftRight = QtGui.QAction(
            text="Flip Mask Left-Right",
            parent=parent,
            icon=QtGui.QIcon(
                Path(Path(parent.data.app_path), "resources", "flipLR.png").__str__()
            ),
        )
        parent.maskFlipLeftRight.setEnabled(False)
        parent.maskFlipLeftRight.triggered.connect(
            lambda x: MenuBar._mask_flip_left_right(parent)
        )
        orientation_menu.addAction(parent.maskFlipLeftRight)

        # Flip Back Forth

        parent.maskFlipBackForth = QtGui.QAction(
            text="Flip Mask Back-Forth",
            parent=parent,
            icon=QtGui.QIcon(
                Path(Path(parent.data.app_path), "resources", "flipZ.png").__str__()
            ),
        )
        parent.maskFlipBackForth.setEnabled(False)
        parent.maskFlipBackForth.triggered.connect(
            lambda x: MenuBar._mask_flip_back_forth(parent)
        )
        orientation_menu.addAction(parent.maskFlipBackForth)

        mask_menu.addMenu(orientation_menu)

        # Mask to Image

        parent.mask2img = QtGui.QAction(text="&Apply on Image", parent=parent)
        parent.mask2img.setEnabled(False)
        parent.mask2img.triggered.connect(lambda x: MenuBar._mask2img(parent))
        mask_menu.addAction(parent.mask2img)

        padding_menu = QtWidgets.QMenu("Zero-Padding", parent)
        parent.pad_image = QtGui.QAction(text="For Image", parent=parent)
        parent.pad_image.setEnabled(True)
        parent.pad_image.triggered.connect(parent.data.nii_img.zero_padding)
        padding_menu.addAction(parent.pad_image)
        parent.pad_seg = QtGui.QAction(text="For Segmentation", parent=parent)
        parent.pad_seg.setEnabled(True)
        # self.pad_seg.triggerd.connect(self.data.nii_seg.super().zero_padding)
        # self.pad_seg.triggered.connect(lambda x: _mask2img(self))
        padding_menu.addAction(parent.pad_seg)
        edit_menu.addMenu(padding_menu)

        edit_menu.addMenu(mask_menu)

        parent.menuBar().addMenu(edit_menu)

        # ----- Fitting Menu
        fit_menu = QtWidgets.QMenu("&Fitting", parent)
        fit_menu.setEnabled(True)

        parent.fit_NNLS = QtGui.QAction(text="NNLS", parent=parent)
        parent.fit_NNLS.triggered.connect(lambda x: MenuBar._fit(parent, "NNLS"))
        fit_menu.addAction(parent.fit_NNLS)

        parent.fit_multiExp = QtGui.QAction(text="IVIM", parent=parent)
        parent.fit_multiExp.triggered.connect(
            lambda x: MenuBar._fit(parent, "multiExp")
        )
        fit_menu.addAction(parent.fit_multiExp)

        parent.menuBar().addMenu(fit_menu)

        # ----- View Menu
        view_menu = QtWidgets.QMenu("&View", parent)
        image_menu = QtWidgets.QMenu("Switch Image", parent)

        parent.plt_showImg = QtGui.QAction(text="Image", parent=parent)
        parent.plt_showImg.triggered.connect(
            lambda x: MenuBar._switch_image(parent, "Img")
        )
        # image_menu.addAction(self.plt_showImg)

        parent.plt_showMask = QtGui.QAction(text="Mask", parent=parent)
        parent.plt_showMask.triggered.connect(
            lambda x: MenuBar._switch_image(parent, "Mask")
        )
        # image_menu.addAction(self.plt_showMask)

        parent.plt_showMaskedImage = QtGui.QAction("Image with applied Mask")
        parent.plt_showMaskedImage.setEnabled(False)
        parent.plt_showMaskedImage.setCheckable(True)
        parent.plt_showMaskedImage.setChecked(False)
        parent.plt_showMaskedImage.toggled.connect(
            lambda x: MenuBar._plt_show_masked_image(parent)
        )
        image_menu.addAction(parent.plt_showMaskedImage)

        # main_window.plt_showDyn = QtGui.QAction(text="Dynamic", parent=main_window)
        # main_window.plt_showDyn.triggered.connect(
        #     lambda x: _switch_image(main_window, "Dyn")
        # )
        # image_menu.addAction(main_window.plt_showDyn)
        view_menu.addMenu(image_menu)

        parent.plt_show = QtGui.QAction("Show Plot")
        parent.plt_show.setEnabled(True)
        parent.plt_show.setCheckable(True)
        parent.plt_show.triggered.connect(lambda x: MenuBar._plt_show(parent))
        view_menu.addAction(parent.plt_show)
        view_menu.addSeparator()

        parent.plt_DispType_SingleVoxel = QtGui.QAction(
            text="Show Single Voxel Spectrum", parent=parent
        )
        parent.plt_DispType_SingleVoxel.setCheckable(True)
        parent.plt_DispType_SingleVoxel.setChecked(True)
        parent.settings.setValue("plt_disp_type", "single_voxel")
        # main_window.plt_DispType_SingleVoxel.toggled.connect(
        #     lambda x: main_window._switchPlt(main_window, "single_voxel")
        # )
        parent.plt_DispType_SingleVoxel.setEnabled(False)
        view_menu.addAction(parent.plt_DispType_SingleVoxel)

        parent.plt_DispType_SegSpectrum = QtGui.QAction(
            text="Show Segmentation Spectrum", parent=parent
        )
        parent.plt_DispType_SegSpectrum.setCheckable(True)
        # main_window.plt_DispType_SegSpectrum.toggled.connect(
        #     lambda x: main_window._switchPlt(main_window, "seg_spectrum")
        # )
        parent.plt_DispType_SegSpectrum.setEnabled(False)
        view_menu.addAction(parent.plt_DispType_SegSpectrum)
        view_menu.addSeparator()

        parent.img_overlay = QtGui.QAction(text="Show Mask Overlay", parent=parent)

        parent.img_overlay.toggled.connect(lambda x: MenuBar._img_overlay(parent))
        view_menu.addAction(parent.img_overlay)
        parent.menuBar().addMenu(view_menu)

        eval_menu = QtWidgets.QMenu("Evaluation", parent)
        eval_menu.setEnabled(False)
        # menu_bar.addMenu(eval_menu)

    @staticmethod
    def _load_image(main_window, path: Path | str = None):
        if not path:
            path = QtWidgets.QFileDialog.getOpenFileName(
                main_window,
                caption="Open Image",
                directory="data",
                filter="NifTi (*.nii *.nii.gz)",
            )[0]
        if path:
            file = Path(path) if path else None
            main_window.data.nii_img = Nii(file)
            if main_window.data.nii_img.path is not None:
                main_window.data.plt["n_slice"].number = main_window.SliceSlider.value()
                main_window.SliceSlider.setEnabled(True)
                main_window.SliceSlider.setMaximum(
                    main_window.data.nii_img.array.shape[2]
                )
                main_window.SliceSpnBx.setEnabled(True)
                main_window.SliceSpnBx.setMaximum(
                    main_window.data.nii_img.array.shape[2]
                )
                main_window.settings.setValue("img_disp_type", "Img")
                main_window.setup_image()
                main_window.mask2img.setEnabled(
                    True if main_window.data.nii_seg.path else False
                )
                main_window.img_overlay.setEnabled(
                    True if main_window.data.nii_seg.path else False
                )
            else:
                print("Warning no file selected")

    @staticmethod
    def _load_seg(parent):
        path = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            caption="Open Mask Image",
            directory="",
            filter="NifTi (*.nii *.nii.gz)",
        )[0]
        if path:
            file = Path(path)
            parent.data.nii_seg = NiiSeg(file)
            if parent.data.nii_seg:
                parent.data.nii_seg.mask = True
                parent.mask2img.setEnabled(True if parent.data.nii_seg.path else False)
                parent.maskFlipUpDown.setEnabled(True)
                parent.maskFlipLeftRight.setEnabled(True)
                parent.maskFlipBackForth.setEnabled(True)

                parent.img_overlay.setEnabled(
                    True if parent.data.nii_seg.path else False
                )
                parent.img_overlay.setChecked(
                    True if parent.data.nii_seg.path else False
                )
                parent.settings.setValue(
                    "img_disp_overlay", True if parent.data.nii_seg.path else False
                )
                if parent.data.nii_img.path:
                    if (
                        not parent.data.nii_img.array.shape[:3]
                        == parent.data.nii_seg.array.shape[:3]
                    ):
                        print("Warning: Image and segmentation shape do not match!")
                        reshape_seg_dlg = ReshapeSegDlg(
                            parent.data.nii_img, parent.data.nii_seg
                        )
                        result = reshape_seg_dlg.exec()
                        if result == QtWidgets.QDialog.accepted or result:
                            parent.data.nii_seg = reshape_seg_dlg.new_seg
                        else:
                            print(
                                "Warning: Img and segmentation shape missmatch still present!"
                            )
                    parent.setup_image()
        else:
            print("Warning: No file selected")

    @staticmethod
    def _load_dyn(parent):
        path = QtWidgets.QFileDialog.getOpenFileName(
            parent, "Open Dynamic Image", "", "NifTi (*.nii *.nii.gz)"
        )[0]
        if path:
            file = Path(path) if path else None
            parent.data.nii_dyn = Nii(file)
        # if self.settings.value("plt_show", type=bool):
        #     Plotting.show_pixel_spectrum(self.plt_AX, self.plt_canvas, self.data)
        else:
            print("Warning no file selected")

    @staticmethod
    def _save_fit_image(parent):
        file_name = parent.data.nii_img.path
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                parent,
                "Save Fit Image",
                file_name.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        parent.data.nii_dyn.save(file)

    @staticmethod
    def _save_masked_image(parent):
        file_name = parent.data.nii_img.path
        file_name = Path(
            str(file_name).replace(file_name.stem, file_name.stem + "_masked")
        )
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                parent,
                "Save Masked Image",
                file_name.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        parent.data.nii_img_masked.save(file)

    @staticmethod
    def _open_settings_dlg(parent):
        parent.settings_dlg = SettingsDlg(
            parent.settings,
            parent.data.plt
            # SettingsDictionary.get_settings_dict(self.data)
        )
        parent.settings_dlg.exec()
        parent.settings, parent.data = parent.settings_dlg.get_settings_data(
            parent.data
        )
        parent.change_theme()
        parent.setup_image()

    @staticmethod
    def _mask_flip_up_down(parent: MainWindow):
        # Images are rotated 90 degrees so lr and ud are switched
        parent.data.nii_seg.array = np.fliplr(parent.data.nii_seg.array)
        parent.data.nii_seg.calculate_polygons()
        parent.setup_image()

    @staticmethod
    def _mask_flip_left_right(parent: MainWindow):
        # Images are rotated 90 degrees so lr and ud are switched
        parent.data.nii_seg.array = np.flipud(parent.data.nii_seg.array)
        parent.data.nii_seg.calculate_polygons()
        parent.setup_image()

    @staticmethod
    def _mask_flip_back_forth(parent: MainWindow):
        parent.data.nii_seg.array = np.flip(parent.data.nii_seg.array, axis=2)
        parent.data.nii_seg.calculate_polygons()
        parent.setup_image()

    @staticmethod
    def _mask2img(parent):
        parent.data.nii_img_masked = Processing.merge_nii_images(
            parent.data.nii_img, parent.data.nii_seg
        )
        if parent.data.nii_img_masked:
            parent.plt_showMaskedImage.setEnabled(True)
            parent.saveMaskedImage.setEnabled(True)

    @staticmethod
    def _save_image(parent: MainWindow):
        file_name = parent.data.nii_img.path
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                parent,
                "Save Image",
                file_name.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        parent.data.nii_img.save(file)

        # ----- Fitting Procedure

    @staticmethod
    def _fit(parent, model_name: str):
        fit_data = parent.data.fit_data
        dlg_dict = dict()

        if model_name == ("NNLS" or "NNLSreg" or "NNLSregCV"):
            if not isinstance(
                fit_data.fit_params,
                (
                    parameters.NNLSParams
                    or parameters.NNLSregParams
                    or parameters.NNLSregCVParams
                ),
            ):
                if type(fit_data.fit_params) == parameters.Parameters:
                    fit_data.fit_params = parameters.NNLSregParams()
                else:
                    dialog = FitParametersDlg(fit_data.fit_params)
                    result = dialog.exec()
                    if result:
                        # TODO: Is reg even the right thing to use here @JJ
                        fit_data.fit_params = parameters.NNLSregParams()
                    else:
                        return
            fit_data.model_name = "NNLS"
            dlg_dict = FittingDictionaries.get_nnls_dict(fit_data.fit_params)
        elif model_name == ("multiExp" or "IVIM"):
            if not (type(fit_data.fit_params) == parameters.MultiExpParams):
                if type(fit_data.fit_params) == parameters.Parameters:
                    fit_data.fit_params = parameters.MultiExpParams()
                else:
                    dialog = FitParametersDlg(fit_data.fit_params)
                    result = dialog.exec()
                    if result:
                        fit_data.fit_params = parameters.MultiExpParams()
                    else:
                        return
            fit_data.model_name = "multiExp"
            dlg_dict = FittingDictionaries.get_multi_exp_dict(fit_data.fit_params)

        # Launch Dlg
        parent.fit_dlg = FittingDlg(model_name, dlg_dict, fit_data.fit_params)
        parent.fit_dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        parent.fit_dlg.exec()

        # Extract Parameters from dlg dict
        b_values = MenuBar._b_values_from_dict(parent)
        fit_data.fit_params.b_values = b_values
        parent.fit_dlg.dict_to_attributes(fit_data.fit_params)

        fit_data.fit_params.n_pools = parent.settings.value("number_of_pools", type=int)

        if parent.fit_dlg.run:
            if (
                hasattr(fit_data.fit_params, "reg_order")
                and fit_data.fit_params.reg_order == "CV"
            ):
                # TODO: need to change params to CV! @TT
                fit_data.fit_params = parameters.NNLSregCVParams()
                # NOTE @JJ this needs to be the parameters set
                # fit_data.fit_params.model = model.Model.NNLSRegCV()
                parent.fit_dlg.dict_to_attributes(fit_data.fit_params)
            elif (
                hasattr(fit_data.fit_params, "reg_order")
                and fit_data.fit_params.reg_order != "CV"
            ):
                fit_data.fit_params.reg_order = int(fit_data.fit_params.reg_order)
                if fit_data.fit_params.reg_order == 0:
                    fit_data.fit_params = parameters.NNLSParams()
                    parent.fit_dlg.dict_to_attributes(fit_data.fit_params)
            fit_data.fit_params.b_values = b_values

            parent.mainWidget.setCursor(QtCore.Qt.CursorShape.WaitCursor)

            # Prepare Data
            fit_data.img = parent.data.nii_img
            fit_data.seg = parent.data.nii_seg

            if fit_data.fit_params.fit_area == "Pixel":
                fit_data.fit_pixel_wise(
                    multi_threading=parent.settings.value("multithreading", type=bool)
                )
                parent.data.nii_dyn = Nii().from_array(
                    parent.data.fit_data.fit_results.spectrum
                )

            elif fit_data.fit_area == "Segmentation":
                fit_data.fit_segmentation_wise()

            parent.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

            parent.saveFitImage.setEnabled(True)

    @staticmethod
    def _plt_show(parent):
        """Plot Axis show Callback"""
        if not parent.plt_show.isChecked():
            # self.plt_spectrum_canvas.setParent(None)
            # self.plt_spectrum_fig.set_visible(False)
            parent.main_hLayout.removeItem(parent.plt_vLayout)
            parent.settings.setValue("plt_show", True)
        else:
            # self.main_hLayout.addWidget(self.plt_spectrum_canvas)
            parent.main_hLayout.addLayout(parent.plt_vLayout)
            parent.settings.setValue("plt_show", True)
        # self.resizeMainWindow()
        parent.resize_figure_axis()
        parent.resize_canvas_size()
        parent.setup_image()

    @staticmethod
    def _plt_show_masked_image(parent):
        if parent.plt_showMaskedImage.isChecked():
            parent.img_overlay.setChecked(False)
            parent.img_overlay.setEnabled(False)
            parent.settings.setValue("img_disp_overlay", False)
            parent.setup_image()
        else:
            parent.img_overlay.setEnabled(True)
            parent.settings.setValue("img_disp_overlay", True)
            parent.setup_image()

    @staticmethod
    def _switch_image(parent: MainWindow, img_type: str = "Img"):
        """Switch Image Callback"""
        parent.settings.setValue("img_disp_type", img_type)
        parent.setup_image()

    @staticmethod
    def _img_overlay(parent):
        """Overlay Callback"""
        parent.settings.setValue(
            "img_disp_overlay", True if parent.img_overlay.isChecked() else False
        )
        parent.setup_image()

    @staticmethod
    def _b_values_from_dict(parent):
        b_values = parent.fit_dlg.fit_dict.pop("b_values", False).value
        if b_values:
            if type(b_values) == str:
                b_values = np.fromstring(
                    b_values.replace("[", "").replace("]", ""), dtype=int, sep="  "
                )
                if b_values.shape != parent.data.fit_data.fit_params.b_values.shape:
                    b_values = np.reshape(
                        b_values, parent.data.fit_data.fit_params.b_values.shape
                    )
            elif type(b_values) == list:
                b_values = np.array(b_values)

            return b_values
        else:
            return parent.data.fit_data.fit_params.b_values
