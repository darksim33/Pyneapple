from __future__ import annotations
import os
import sys

from multiprocessing import freeze_support
from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path

from nifti import Nii

from .dlg_fitting import FittingDlg
from .dlg_settings import SettingsDlg
from .appdata import AppData
from .canvas_image import ImageCanvas
from .canvas_plot import PlotLayout
from .context_menu_canvas import (
    PlotDecayContextMenu,
    ImageContextMenu,
)
from .menubar_builder import MenubarBuilder
from .menu_file import FileMenu
from .menu_edit import EditMenu
from .menu_view import ViewMenu
from .menu_fitting import FittingMenu
from .eventfilter import Filter

# v0.7.1


# noinspection PyUnresolvedReferences
class MainWindow(QtWidgets.QMainWindow):
    file_menu: FileMenu
    edit_menu: EditMenu
    fitting_menu: FittingMenu
    view_menu: ViewMenu
    image_axis: ImageCanvas
    plot_layout: PlotLayout

    def __init__(self, path: Path | str = None) -> None:
        """The Main App Window."""
        super(MainWindow, self).__init__()

        self.data = AppData()

        # For Debugging
        self.data.last_dir = Path("")
        self.fit_dlg = FittingDlg
        self.settings_dlg = SettingsDlg

        # Load Settings
        self._load_settings()
        # Set up UI
        self._setup_ui()

        if path:
            print("Path passed!")
        #     self._load_image(path)

    def _load_settings(self):
        """
        The _load_settings function is used to load the settings from a QSettings object.

        The QSettings object is initialized with the application name and organization name,
        which are both &quot;PyNeapple&quot;. The last_dir setting is set to the directory of this file,
        and if it does not exist in self.settings then it will be created as an empty string.
        The theme setting will be set to &quot;Light&quot, if it does not already exist in self.settings.
        """

        self.settings = QtCore.QSettings("MyApp", "PyNeapple")
        if self.settings.value("last_dir", "") == "":
            self.settings.setValue("last_dir", os.path.abspath(__file__))
            self.settings.setValue("theme", "Light")  # "Dark", "Light"
        self.settings.setValue("plt_show", False)

        if not self.settings.contains("number_of_pools"):
            self.settings.setValue("number_of_pools", 4)

        if not self.settings.contains("default_seg_colors"):
            self.settings.setValue(
                "default_seg_colors", ["#ff0000", "#0000ff", "#00ff00", "#ffff00"]
            )
        self.data.plt["seg_colors"] = self.settings.value(
            "default_seg_colors", type=list
        )

        if not self.settings.contains("default_seg_edge_alpha"):
            self.settings.setValue("default_seg_edge_alpha", 0.8)
        self.data.plt["seg_edge_alpha"] = self.settings.value(
            "default_seg_edge_alpha", type=float
        )

        if not self.settings.contains("default_seg_line_width"):
            self.settings.setValue("default_seg_line_width", 2.0)
        self.data.plt["seg_line_width"] = self.settings.value(
            "default_seg_line_width", type=float
        )

        if not self.settings.contains("default_seg_face_alpha"):
            self.settings.setValue("default_seg_face_alpha", 0.0)
        self.data.plt["seg_face_alpha"] = self.settings.value(
            "default_seg_face_alpha", type=float
        )

        if not self.settings.contains("multithreading"):
            self.settings.setValue("multithreading", True)

        if not self.settings.contains("plt_disp_type"):
            self.settings.setValue("plt_disp_type", "single_voxel")

    def _setup_ui(self):
        """Setup Main Window UI"""
        # ----- Window setting
        self.setMinimumSize(512, 512)
        self.setWindowTitle("PyNeapple")
        img = (self.data.app_path / "resources" / "images" / "app.ico").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        self.mainWidget = QtWidgets.QWidget()

        # ----- Menubar
        _ = MenubarBuilder(self)
        # create_menu_bar(self)
        # menubar = MenuBar
        # menubar.setup_menubar(parent=self)

        # ----- Main vertical Layout
        self.main_hLayout = QtWidgets.QHBoxLayout()  # Main horizontal Layout

        self.image_axis = ImageCanvas(
            self,
            self.data.nii_img,
            self.data.nii_seg,
            self.data.plt,
            self.width(),
            self.settings.value("theme", str),
        )
        self.image_axis.deploy_event("button_press_event", self.event_filter_image)

        self.main_hLayout.addLayout(self.image_axis)
        self.mainWidget.setLayout(self.main_hLayout)
        self.setCentralWidget(self.mainWidget)

        # ----- Plotting Frame
        self.plot_layout = PlotLayout(self.data)
        self.plot_layout.decay.deploy_event(
            "button_press_event", self.event_filter_plot_decay
        )
        # self.main_hLayout.addLayout(self.plot_layout)

        # ----- Context Menu
        self.context_menu_image = ImageContextMenu(self)
        self.context_menu_plot_decay = PlotDecayContextMenu(self)

        # ----- StatusBar
        # self.statusBar = QtWidgets.QStatusBar()
        # self.setStatusBar(self.statusBar)

    # Events
    def event_filter_image(self, event):
        Filter.event_filter_image_canvas(self, event)

    def event_filter_plot_decay(self, event):
        Filter.event_filter_plot_decay(self, event)

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

    def change_theme(self):
        """Changes the App theme"""
        theme = self.settings.value("theme")
        if theme == "Dark":
            if "Fusion" in QtWidgets.QStyleFactory.keys():
                QtWidgets.QApplication.setStyle("Fusion")
            else:
                "Warning: No suitable theme found!"
        elif theme == "Light":
            if "windowsvista" in QtWidgets.QStyleFactory.keys():
                QtWidgets.QApplication.setStyle("windowsvista")
            elif "Windows" in QtWidgets.QStyleFactory.keys():
                QtWidgets.QApplication.setStyle("Windows")
            else:
                "Warning: No suitable theme found!"
        self.image_axis.theme = theme


def run():
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()  # QtWidgets.QWidget()
    main_window.change_theme()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
