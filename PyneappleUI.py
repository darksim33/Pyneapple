import os
import sys

from multiprocessing import freeze_support

from PyQt6 import QtWidgets, QtGui, QtCore
from pathlib import Path
import numpy as np

from src.ui.fittingdlg import FittingDlg
from src.ui.settingsdlg import SettingsDlg
from src.utils import Nii
from src.appdata import AppData
from src.ui.menubar.menubarbuilder import MenuBarBuilder
from src.ui.contextmenu import create_context_menu
from src.ui.imagecanvas import ImageCanvas
from src.ui.plotcanvas import PlotLayout
from src.ui.menubar.filemenu import FileMenu
from src.ui.menubar.editmenu import EditMenu
from src.ui.menubar.fittingmenu import FittingMenu
from src.ui.menubar.viewmenu import ViewMenu

# v0.5.1


# noinspection PyUnresolvedReferences
class MainWindow(QtWidgets.QMainWindow):
    file_menu: FileMenu
    edit_menu: EditMenu
    fitting_menu: FittingMenu
    view_menu: ViewMenu
    image_axis: ImageCanvas
    plot_layout: PlotLayout

    def __init__(self, path: Path | str = None) -> None:
        """
        The Main App Window.
        """
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
        """
        The _load_settings function is used to load the settings from a QSettings object.
        The QSettings object is initialized with the application name and organization name,
        which are both &quot;Pyneapple&quot;. The last_dir setting is set to the directory of this file,
        and if it does not exist in self.settings then it will be created as an empty string.
        The theme setting will be set to &quot;Light&quot; if it does not already exist in self.settings.
        """

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
        """Setup Main Window UI"""
        # ----- Window setting
        self.setMinimumSize(512, 512)
        self.setWindowTitle("Pyneapple")
        img = Path(Path(__file__).parent, "resources", "PyneappleLogo.ico").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        self.mainWidget = QtWidgets.QWidget()

        # ----- Menubar
        _ = MenuBarBuilder(self)
        # create_menu_bar(self)
        # menubar = MenuBar
        # menubar.setup_menubar(parent=self)
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
        self.plot_layout = PlotLayout(self.data)
        # self.main_hLayout.addLayout(self.plot_layout)

        # ----- StatusBar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

    # Events
    def event_filter(self, event):
        """
        Event Filter Handler.

        The event_filter function is used to filter events that are passed to the
        event_handler. This function is called by the event handler and should return
        True if it wants the event handler to process this event, or False if it wants
        the event handler to ignore this particular mouse click. The default behavior of
        this function is always returning True, which means all mouse clicks will be processed.

        Parameters
        ----------
            self
                Refer to the current instance of a class
            event
                Get the mouse click position
        Returns
        -------
            A boolean value

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
                            self.plot_layout.data = self.data
                            self.plot_layout.plot_pixel_decay(position)

                            if np.any(self.data.nii_dyn.array):
                                self.plot_layout.plot_pixel_fit(position)
                                self.plot_layout.plot_pixel_spectrum(position)
                        # elif (
                        #     self.settings.value("plt_disp_type", type=str)
                        #     == "seg_spectrum"
                        # ):
                        #     plotting.show_seg_spectrum(
                        #         self.plt_spectrum_AX,
                        #         self.plt_spectrum_canvas,
                        #         self.data,
                        #         0,
                        #     )

    def contextMenuEvent(self, event):
        """Context Menu Event"""
        self.context_menu.popup(QtGui.QCursor.pos())

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
