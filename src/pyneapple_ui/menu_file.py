from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
from PyQt6 import QtWidgets
from PyQt6.QtGui import QAction, QIcon
from typing import TYPE_CHECKING

from .dlg_prompts import (
    ReshapeSegMessageBox,
    AlreadyLoadedSegMessageBox,
    StillLoadedSegMessageBox,
)
from .dlg_settings import SettingsDlg
from nifti import Nii, NiiSeg
from .appdata import AppData

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class LoadFileAction(QAction):
    def __init__(self, parent: MainWindow, text: str, icon: QIcon | str):
        """
        Basic QAction to handle file imports (abstract).

        It sets up the class with all of its attributes and methods.
        The self parameter refers to the instance of this object, which will be created in MainWindow.

        Parameters
        ----------
            self
                Refer to the current instance of a class
            parent: MainWindow
                Pass the main window object to the class
            text: str
                Set the text of the menu item
            icon: QIcon | str
                Set the icon of the button
        """
        super().__init__()
        self.parent = parent
        self.setText(text)
        self.setIcon(icon)
        self.triggered.connect(self.load)

    @abstractmethod
    def load(self):
        """Load function that is executed on trigger."""
        pass


class LoadImageAction(LoadFileAction):
    def __init__(self, parent: MainWindow):
        """
        Action to load a NifTi image.

        It sets up the instance of the class, and makes sure that it has all
        the attributes necessary for proper functioning.  The __init__ function
        is also responsible for setting up inheritance, if any.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the main window object to this class
        """
        super().__init__(
            parent,
            "Open &Image...",
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "images",
                    "app.ico",
                ).__str__()
            ),
        )

    def load(self, file_path: Path | None = None):
        """
        Load NifTii Image.

        The load function is called when the user clicks on the &quot;Load Image&quot; button.
        It opens a file dialog and allows the user to select an image file. The selected image
        is then loaded into memory and displayed in the main window.

        Parameters
        ----------
            self
                Refer to the current instance of a class
            file_path: Path | None
                Specify that the path parameter can be either a path object or none
        Returns
        -------
            The image that was loaded

        """
        # Check if there already is a Seg loaded when changing img
        if self.parent.data.nii_seg.path:
            prompt = AlreadyLoadedSegMessageBox()
            if prompt.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                self.parent.data.nii_seg.clear()
                self.parent.image_axis.segmentation.clear()
        if not file_path:
            file_path = QtWidgets.QFileDialog.getOpenFileName(
                self.parent,
                caption="Open Image",
                directory=self.parent.data.last_dir.__str__(),
                filter="NifTi (*.nii *.nii.gz)",
            )[0]
            self.parent.data.last_dir = Path(file_path).parent
        if file_path:
            # Clear Image Axis if necessary
            if self.parent.data.nii_img.path:
                self.parent.image_axis.clear()
            # Load File
            file_path = Path(file_path) if file_path else None
            self.parent.data.nii_img = Nii(file_path)
            if self.parent.data.nii_img.path is not None:
                # UI handling
                self.parent.settings.setValue("img_disp_type", "Img")
                self.parent.edit_menu.seg2img.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.view_menu.show_seg_overlay.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                # display image
                self.parent.image_axis.image = self.parent.data.nii_img
            else:
                print("Warning no file selected")


class LoadSegAction(LoadFileAction):
    def __init__(self, parent: MainWindow):
        """
        Action to load a Segmentation from a NifTi file.

        It allows the class to initialize the attributes of a class.
        The self parameter refers to the instance of an object, and is used to access variables that belongs to a
        specific instance.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the class
        """
        super().__init__(
            parent,
            "Open &Segmentation...",
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "images",
                    "load_seg.ico",
                ).__str__()
            ),
        )

    def load(self):
        """
        Loads Segmentation.

        Opens a file dialog and allows the user to select an image file. The selected
        file is then loaded as a NiiSeg object. If this function was
        called from within another function, it would return this NiiSeg object.
        """

        # Check if there still is a Seg loaded when loading in new one
        if self.parent.data.nii_seg.path:
            prompt = StillLoadedSegMessageBox()
            if prompt.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                self.parent.data.nii_seg.clear()

        file_path = QtWidgets.QFileDialog.getOpenFileName(
            self.parent,
            caption="Open Segmentation Image",
            directory=self.parent.data.last_dir.__str__(),
            filter="NifTi (*.nii *.nii.gz)",
        )[0]
        self.parent.data.last_dir = Path(file_path).parent
        if file_path:
            # Load File
            file_path = Path(file_path)
            self.parent.data.nii_seg = NiiSeg(file_path)
            if self.parent.data.nii_seg:
                # UI handling
                self.parent.data.nii_seg.mask = True
                self.parent.edit_menu.seg2img.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.edit_menu.seg_flip_up_down.setEnabled(True)
                self.parent.edit_menu.seg_flip_left_right.setEnabled(True)
                self.parent.edit_menu.seg_flip_back_forth.setEnabled(True)

                self.parent.view_menu.show_seg_overlay.setEnabled(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.view_menu.show_seg_overlay.setChecked(
                    True if self.parent.data.nii_seg.path else False
                )
                self.parent.settings.setValue(
                    "img_disp_overlay",
                    True if self.parent.data.nii_seg.path else False,
                )
                if self.parent.data.nii_img.path:
                    # Reshaping Segmentation if needed
                    if (
                        not self.parent.data.nii_img.array.shape[:3]
                        == self.parent.data.nii_seg.array.shape[:3]
                    ):
                        print("Warning: Image and segmentation shape do not match!")
                        reshape_seg_dlg = ReshapeSegMessageBox()
                        if (
                            reshape_seg_dlg.exec()
                            == QtWidgets.QMessageBox.StandardButton.Yes
                        ):
                            self.parent.data.nii_seg = reshape_seg_dlg.reshape(
                                self.parent.data.nii_img, self.parent.data.nii_seg
                            )
                        else:
                            print(
                                "Warning: Img and segmentation shape missmatch still present!"
                            )
                    self.parent.image_axis.segmentation = self.parent.data.nii_seg
        else:
            print("Warning: No file selected")
        self.parent.image_axis.setup_image()


class LoadDynamicAction(LoadFileAction):
    def __init__(self, parent: MainWindow):
        """
        Load dynamic (spectral) NifTi image Action.

        The __init__ function is called when the class is instantiated.
        It sets up the menu item's text, icon, and shortcut key.

        Parameters
        ----------
            self
                Represent the instance of the object itself
            parent: MainWindow
                Pass the parent window to the class
        """
        super().__init__(
            parent,
            "Open &Dynamic Image...",
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "images",
                    "PineappleLogo_Dyn.png",
                ).__str__()
            ),
        )

    def load(self):
        """
        Load dynamic image.

        The _load_dyn function is a helper function that opens the file dialog and
        loads the selected dynamic image into the data object. It also updates
        the plot if it is enabled.
        """
        file_path = QtWidgets.QFileDialog.getOpenFileName(
            self.parent,
            caption="Open Dynamic Image",
            directory=self.parent.data.last_dir.__str__(),
            filter="NifTi (*.nii *.nii.gz)",
        )[0]
        self.parent.data.last_dir = Path(file_path).parent
        if file_path:
            file_path = Path(file_path) if file_path else None
            self.parent.data.nii_dyn = Nii(file_path)
        # if self.settings.value("plt_show", type=bool):
        #     Plotting.show_pixel_spectrum(self.plt_AX, self.plt_canvas, self.data)
        else:
            print("Warning no file selected")


class ClearImageAction(QAction):
    def __init__(self, parent: MainWindow):
        """
        Clear the current loaded image from App Action.

        It sets up the class object with all of its attributes and methods.
        The self parameter refers to the instance of this class that has been created.

        Parameters
        ----------
            self
                Refer to the object itself
            parent: MainWindow
                Pass the parent window to the class
        """
        super().__init__()
        self.parent = parent
        self.setText("Clear Image")
        self.setIcon(
            QIcon(
                Path(
                    Path(parent.data.app_path),
                    "resources",
                    "images",
                    "clear_img.ico",
                ).__str__()
            )
        )
        self.triggered.connect(self.clear)

    def clear(self):
        """
        Remove image from App and clear Axis.

        The clear function clears the image axis and resets the data to an empty AppData object.

        Parameters
        ----------
            self
                Access the attributes of the class
        """
        self.parent.image_axis.clear()
        self.parent.plot_layout.clear()
        self.parent.data = AppData()
        print("Cleared")


class SaveFileAction(QAction):
    def __init__(self, parent: MainWindow, text: str, icon: QIcon):
        """
        Basic save file Action (abstract).

        Basic class for further inheritance to save files.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the class
            text: str
                Set the text of the menu item
            icon: QIcon
                Set the icon of the button
        """
        super().__init__()
        self.parent = parent
        self.setText(text)
        self.setIcon(icon)
        self.triggered.connect(self.save)

    @abstractmethod
    def save(self):
        """Save function. Needs to be deployed."""
        pass


class SaveImageAction(SaveFileAction):
    def __init__(self, parent: MainWindow):
        """Action to save the currently loaded image to a NifTi file."""
        super().__init__(
            parent,
            "Save Image...",
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )

    def save(self):
        """
        Save the currently loaded image to a NifTi file.

        The save function is used to save the image that has been created.
        It will open a file dialog box and allow you to choose where you want
        to save the image. It will then save it as a NifTi file.
        """
        file_name = self.parent.data.nii_img.path
        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                caption="Save Image",
                directory=self.parent.data.last_dir.__str__(),
                filter="NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent
        self.parent.data.nii_img.save(file_path)


class SaveFitImageAction(SaveFileAction):
    def __init__(self, parent: MainWindow):
        """Action to save the currently processed fit spectrum to a 4D-NifTi file."""
        super().__init__(
            parent,
            "Save Fit to NifTi...",
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.setEnabled(False)

    def save(self):
        """Save the currently processed fit spectrum to a 4D-NifTi file."""
        file_name = self.parent.data.nii_img.path
        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                caption="Save Fit Image",
                directory=self.parent.data.last_dir.__str__(),
                filter="NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent
        self.parent.data.nii_dyn.save(file_path)


class SaveSegmentedImageAction(SaveFileAction):
    def __init__(self, parent: MainWindow):
        """Action to save the image with applied mask if created."""
        super().__init__(
            parent,
            "Save Masked Image...",
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.setEnabled(False)

    def save(self):
        """Save the image with applied mask if created."""
        file_name = self.parent.data.nii_img.path
        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                caption="Save Masked Image",
                directory=(
                    self.parent.data.last_dir / (file_name.stem + "_masked.nii.gz")
                ).__str__(),
                filter="NifTi (*.nii *.nii.gz)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent
        self.parent.data.nii_img_masked.save(file_path)


class OpenSettingsAction(QAction):
    def __init__(self, parent: MainWindow):
        """Action to open Settings Dialog."""
        super().__init__()
        self.parent = parent
        self.setText("Settings...")
        self.setIcon(
            QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "images", "settings.ico"
                ).__str__()
            )
        )
        self.triggered.connect(self.open)

    def open(self):
        """Open Settings Dialog."""
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
    save_segmented_image: SaveSegmentedImageAction
    open_settings: OpenSettingsAction

    def __init__(self, parent: MainWindow):
        """
        QMenu to handle the basic file and app related actions.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the menu
        """
        super().__init__("&File", parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Sets up menu."""
        # Load Image
        self.load_image = LoadImageAction(self.parent)
        self.addAction(self.load_image)
        self.load_seg = LoadSegAction(self.parent)
        self.addAction(self.load_seg)
        self.load_dyn = LoadDynamicAction(self.parent)
        # self.addAction(self.load_dyn)
        self.clear_image = ClearImageAction(self.parent)
        self.addAction(self.clear_image)
        self.addSeparator()  # -----------------------
        self.save_image = SaveImageAction(self.parent)
        self.addAction(self.save_image)
        self.save_fit_image = SaveFitImageAction(self.parent)
        # self.addAction(self.save_fit_image)
        self.save_segmented_image = SaveSegmentedImageAction(self.parent)
        self.addAction(self.save_segmented_image)
        self.addSeparator()  # -----------------------
        self.open_settings = OpenSettingsAction(self.parent)
        self.addAction(self.open_settings)
