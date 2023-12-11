from __future__ import annotations
from abc import abstractmethod
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtGui import QAction, QIcon

from typing import TYPE_CHECKING, Type

from src.ui.promptdlgs import (
    ReshapeSegDlg,
    AlreadyLoadedSegDlg,
    StillLoadedSegDlg,
)
from src.ui.settingsdlg import SettingsDlg
from src.utils import Nii, NiiSeg
from src.appdata import AppData

if TYPE_CHECKING:
    from PyneappleUI import MainWindow


class LoadFileAction(QAction):
    def __init__(self, parent: MainWindow, text: str, icon: QIcon | str):
        super().__init__()
        self.parent = parent
        self.setText(text)
        self.setIcon(icon)
        self.triggered.connect(self.load)

    @abstractmethod
    def load(self):
        pass


class LoadImageAction(LoadFileAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            parent,
            "Open &Image...",
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "PineappleLogo.png",
                ).__str__()
            ),
        )

    def load(self, path: Path | None = None):
        """
        The load function is called when the user clicks on the &quot;Load Image&quot; button.
        It opens a file dialog and allows the user to select an image file. The selected image
        is then loaded into memory and displayed in the main window.

        Parameters
        ----------
            self
                Refer to the current instance of a class
            path: Path | None
                Specify that the path parameter can be either a path object or none

        Returns
        -------

            The image that was loaded

        Doc Author
        ----------
            Trelent
        """
        # Check if there already is a Seg loaded when changing img
        if self.parent.data.nii_seg.path:
            prompt = AlreadyLoadedSegDlg()
            result = prompt.exec()
            if not result:
                self.parent.data.nii_seg = Nii()

        if not path:
            path = QtWidgets.QFileDialog.getOpenFileName(
                self.parent,
                caption="Open Image",
                directory="data",
                filter="NifTi (*.nii *.nii.gz)",
            )[0]
        if path:
            # Load File
            file = Path(path) if path else None
            self.parent.data.nii_img = Nii(file)
            if self.parent.data.nii_img.path is not None:
                # UI handling
                self.parent.settings.setValue("img_disp_type", "Img")
                self.parent.mask2img.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.img_overlay.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                # display image
                self.parent.image_axis.image = self.parent.data.nii_img
            else:
                print("Warning no file selected")


class LoadSegAction(LoadFileAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            parent,
            "Open &Segmentation...",
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "PineappleLogo_Seg.png",
                ).__str__()
            ),
        )

    def load(self):
        """
        Opens a file dialog and allows the user to select an image file. The selected
        file is then loaded as a NiiSeg object. If this function was
        called from within another function, it would return this NiiSeg object.
        """

        # Check if there still is a Seg loaded when loading in new one
        if self.parent.data.nii_seg.path:
            prompt = StillLoadedSegDlg()
            result = prompt.exec()
            if not result:
                self.parent.data.nii_seg = Nii()

        path = QtWidgets.QFileDialog.getOpenFileName(
            self.parent,
            caption="Open Mask Image",
            directory="",
            filter="NifTi (*.nii *.nii.gz)",
        )[0]
        if path:
            # Load File
            file = Path(path)
            self.parent.data.nii_seg = NiiSeg(file)
            if self.parent.data.nii_seg:
                # UI handling
                self.parent.data.nii_seg.mask = True  # FIXME: necessary?
                self.parent.mask2img.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.maskFlipUpDown.setEnabled(True)
                self.parent.maskFlipLeftRight.setEnabled(True)
                self.parent.maskFlipBackForth.setEnabled(True)

                self.parent.img_overlay.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.img_overlay.setChecked(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.settings.setValue(
                    "img_disp_overlay",
                    True if self.parent.data.nii_seg.path else False,
                )  # FIXME: always on???
                if self.parent.data.nii_img.path:
                    # Reshaping Segmentation if needed
                    if (
                        not self.parent.data.nii_img.array.shape[:3]
                        == self.parent.data.nii_seg.array.shape[:3]
                    ):
                        print("Warning: Image and segmentation shape do not match!")
                        reshape_seg_dlg = ReshapeSegDlg(
                            self.parent.data.nii_img,
                            self.parent.data.nii_seg,
                        )
                        result = reshape_seg_dlg.exec()
                        if result == QtWidgets.QDialog.accepted or result:
                            self.parent.data.nii_seg = reshape_seg_dlg.new_seg
                        else:
                            print(
                                "Warning: Img and segmentation shape missmatch still present!"
                            )
                    self.parent.image_axis.segmentation = self.parent.data.nii_seg
        else:
            print("Warning: No file selected")


class LoadDynamicAction(LoadFileAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            parent,
            "Open &Dynamic Image...",
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "PineappleLogo_Dyn.png",
                ).__str__()
            ),
        )

    def load(self):
        """
        Load dynamic image callback.

        The _load_dyn function is a helper function that opens the file dialog and
        loads the selected dynamic image into the data object. It also updates
        the plot if it is enabled.

        Parameters
        ----------
            parent
                Pass the parent object to the function
        Returns
        -------
            A nii object
        """
        path = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, "Open Dynamic Image", "", "NifTi (*.nii *.nii.gz)"
        )[0]
        if path:
            file = Path(path) if path else None
            self.parent.data.nii_dyn = Nii(file)
        # if self.settings.value("plt_show", type=bool):
        #     Plotting.show_pixel_spectrum(self.plt_AX, self.plt_canvas, self.data)
        else:
            print("Warning no file selected")


class ClearImageAction(QAction):
    def __init__(self, parent: MainWindow):
        super().__init__()
        self.parent = parent
        self.setText("Clear Image")
        self.setIcon(
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "PineappleLogo_ClearImage.png",
                ).__str__()
            )
        )
        self.triggered.connect(self.clear)

    def clear(self):
        self.parent.image_axis.clear()
        self.parent.data = AppData()
        print("Cleared")


class SaveFileAction(QAction):
    def __init__(self, parent: MainWindow, text: str, icon: QIcon):
        super().__init__()
        self.parent = parent
        self.setText(text)
        self.setIcon(icon)
        self.triggered.connect(self.save)

    @abstractmethod
    def save(self):
        pass


class SaveImageAction(SaveFileAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            parent,
            "Save Image...",
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )

    def save(self):
        file_name = self.parent.data.nii_img.path
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Save Image",
                file_name.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.parent.data.nii_img.save(file)


class SaveFitImageAction(SaveFileAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            parent,
            "Save Fit to NifTi...",
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )

    def save(self):
        file_name = self.parent.data.nii_img.path
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Save Fit Image",
                file_name.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.parent.data.nii_dyn.save(file)


class SaveMaskedImageAction(SaveFileAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            parent,
            "Save Masked Image...",
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )

    def save(self):
        file_name = self.parent.data.nii_img.path
        file_name = Path(
            str(file_name).replace(file_name.stem, file_name.stem + "_masked")
        )
        file = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Save Masked Image",
                file_name.__str__(),
                "NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.parent.data.nii_img_masked.save(file)


class OpenSettingsAction(QAction):
    def __init__(self, parent: MainWindow):
        super().__init__()
        self.parent = parent
        self.setText("Settings...")
        self.setIcon(
            QIcon(
                Path(Path(parent.data.app_path), "resources", "Settings.ico").__str__()
            )
        )
        self.triggered.connect(self.open)

    def open(self):
        self.parent.settings_dlg = SettingsDlg(
            self.parent.settings,
            self.parent.data.plt
            # SettingsDictionary.get_settings_dict(self.data)
        )
        self.parent.settings_dlg.exec()
        (
            self.parent.settings,
            self.parent.data,
        ) = self.parent.settings_dlg.get_settings_data(self.parent.data)
        self.parent.change_theme()


class FileMenu(QtWidgets.QMenu):
    load_image: QAction
    load_seg: LoadSegAction
    clear_image: ClearImageAction
    load_dyn: LoadDynamicAction
    save_image: SaveImageAction
    save_fit_image: SaveFitImageAction
    save_masked_image: SaveMaskedImageAction
    open_settings: OpenSettingsAction

    def __init__(self, parent: MainWindow):
        super().__init__("&File", parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        # Load Image
        self.load_image = LoadImageAction(self.parent)
        self.addAction(self.load_image)
        self.load_seg = LoadSegAction(self.parent)
        self.addAction(self.load_seg)
        self.load_dyn = LoadDynamicAction(self.parent)
        self.addAction(self.load_dyn)
        self.clear_image = ClearImageAction(self.parent)
        self.addAction(self.clear_image)
        self.addSeparator()  # -----------------------
        self.save_image = SaveImageAction(self.parent)
        self.addAction(self.save_image)
        self.save_fit_image = SaveFitImageAction(self.parent)
        self.addAction(self.save_fit_image)
        self.save_masked_image = SaveMaskedImageAction(self.parent)
        self.addAction(self.save_masked_image)
        self.addSeparator()  # -----------------------
        self.open_settings = OpenSettingsAction(self.parent)
        self.addAction(self.open_settings)
