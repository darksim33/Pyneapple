import os
import sys

from multiprocessing import freeze_support

from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
from PIL import Image
import numpy as np

import src.plotting as plotting
from src.ui.fittingdlg import FittingDlg
from src.ui.settingsdlg import SettingsDlg
from src.utils import Nii
from src.appdata import AppData
from src.ui.menubar import MenuBar
from src.ui.contextmenu import create_context_menu

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

            self.settings.setValue("default_seg_alpha", 0.5)
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

        if not self.settings.contains("default_seg_alpha"):
            self.settings.setValue("default_seg_alpha", 0.5)
        self.data.plt["seg_alpha"] = self.settings.value(
            "default_seg_alpha", type=float
        )

        if not self.settings.contains("default_seg_line_width"):
            self.settings.setValue("default_seg_line_width", 2.0)
        self.data.plt["seg_line_width"] = self.settings.value(
            "default_seg_line_width", type=float
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
        theme = self.settings.value("theme", type=str)
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
        self.SliceHLayout = QtWidgets.QHBoxLayout()  # Layout for Slider ans Spinbox
        self.SliceSlider = QtWidgets.QSlider()
        self.SliceSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.SliceSlider.setEnabled(False)
        self.SliceSlider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.SliceSlider.setTickInterval(1)
        self.SliceSlider.setMinimum(1)
        self.SliceSlider.setMaximum(20)
        self.SliceSlider.valueChanged.connect(lambda x: self._slice_slider_changed())
        self.SliceHLayout.addWidget(self.SliceSlider)

        # ----- SpinBox
        self.SliceSpnBx = QtWidgets.QSpinBox()
        self.SliceSpnBx.setValue(1)
        self.SliceSpnBx.setEnabled(False)
        self.SliceSpnBx.setMinimumWidth(20)
        self.SliceSpnBx.setMaximumWidth(40)
        self.SliceSpnBx.valueChanged.connect(lambda x: self._slice_spn_bx_changed())
        self.SliceHLayout.addWidget(self.SliceSpnBx)

        self.main_vLayout.addLayout(self.SliceHLayout)

        # Adjust Canvas and Slider size according to main window
        self.resize_canvas_size()
        self.resize_figure_axis()
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

    def _slice_slider_changed(self):
        """Slice Slider Callback"""
        self.data.plt["n_slice"].number = self.SliceSlider.value()
        self.SliceSpnBx.setValue(self.SliceSlider.value())
        self.setup_image()

    def _slice_spn_bx_changed(self):
        """Slice Spinbox Callback"""
        self.data.plt["n_slice"].number = self.SliceSpnBx.value()
        self.SliceSlider.setValue(self.SliceSpnBx.value())
        self.setup_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_canvas_size()
        self.resize_figure_axis()
        self.setup_image()

    def changeEvent(self, event):
        # Override the change event handler
        if type(event) == QtGui.QWindowStateChangeEvent:
            self.resize_canvas_size()
            self.resize_figure_axis()
            self.setup_image()

    def resize_figure_axis(self, aspect_ratio: tuple | None = (1.0, 1.0)):
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

    def resize_canvas_size(self):
        if self.settings.value("plt_show", type=bool):
            # canvas_size = self.img_canvas.size()
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
                                if self.data.plt["seg_face"]:
                                    polygon_patch.set_facecolor(colors[seg_color_idx])
                                else:
                                    polygon_patch.set_facecolor("none")
                                self.img_ax.add_patch(polygon_patch)
                        seg_color_idx += 1

            self.img_ax.axis("off")
            self.resize_canvas_size()
            self.resize_figure_axis()
            self.img_canvas.draw()

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
