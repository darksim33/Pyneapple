import os
import sys

from multiprocessing import freeze_support

from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
from PIL import Image
import numpy as np
from copy import deepcopy

import src.plotting as plotting
from src.ui.fittingdlg import FittingDlg
from src.ui.settingsdlg import SettingsDlg
from src.utils import Nii
from src.appdata import AppData
from src.ui.menubar import MenuBar
from src.ui.contextmenu import create_context_menu
from src.ui.imagecanvas import ImageCanvas

# v0.5.1


# noinspection PyUnresolvedReferences
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: Path | str = None) -> None:
        super(MainWindow, self).__init__()

        self.data = AppData()
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
        self.settings = QtCore.QSettings("MyApp", "Pyneapple")
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

    def _setup_ui(self):
        # ----- Window setting
        self.setMinimumSize(512, 512)
        self.setWindowTitle("Pyneapple")
        img = Path(Path(__file__).parent, "resources", "PineappleLogo.png").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        self.mainWidget = QtWidgets.QWidget()

        # ----- Menubar
        # create_menu_bar(self)
        menubar = MenuBar
        menubar.setup_menubar(parent=self)
        # ----- Context Menu
        create_context_menu(self)

        # ----- Main vertical Layout
        self.main_hLayout = QtWidgets.QHBoxLayout()  # Main horizontal Layout

        self.image_axis = ImageCanvas(
            self.data.nii_img,
            self.data.nii_seg,
            self.data.plt,
            self.width(),
            self.settings.value("theme", str),
        )
        self.image_axis.deploy_event("button_press_event", self.event_filter)

        self.main_hLayout.addLayout(self.image_axis)
        self.mainWidget.setLayout(self.main_hLayout)
        self.setCentralWidget(self.mainWidget)

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

        # self.main_hLayout.addLayout(self.main_vLayout)

        # ----- StatusBar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

    # Events
    def event_filter(self, event):
        """
        The event_filter function is used to filter events that are passed to the
        event_handler. This function is called by the event handler and should return
        True if it wants the event handler to process this event, or False if it wants
        the event handler to ignore this particular mouse click. The default behavior of
        this function is always returning True, which means all mouse clicks will be processed.

        :param self: Refer to the class itself
        :param event: Get the position of the mouse click
        :return: The position of the mouse click on the image
        :doc-author: Trelent
        """
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
                                plotting.show_pixel_fit(
                                    self.plt_signal_AX,
                                    self.plt_signal_canvas,
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

    # def resizeEvent(self, event):
    #     super().resizeEvent(event)
    #     self.resize_canvas_size()
    #     self.resize_figure_axis()
    #     self.setup_image()
    #
    # def changeEvent(self, event):
    #     # Override the change event handler
    #     if type(event) == QtGui.QWindowStateChangeEvent:
    #         self.resize_canvas_size()
    #         self.resize_figure_axis()
    #         self.setup_image()

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
        theme = self.settings.value("theme")
        if theme == "Dark":
            QtWidgets.QApplication.setStyle("Fusion")
        elif theme == "Light":
            QtWidgets.QApplication.setStyle("windowsvista")
        self.image_axis.theme = theme


if __name__ == "__main__":
    freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()  # QtWidgets.QWidget()
    main_window.change_theme()
    main_window.show()
    sys.exit(app.exec())
