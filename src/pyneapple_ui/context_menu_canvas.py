from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6 import QtWidgets, QtGui
from pathlib import Path
import numpy as np

from .menu_view import ShowPlotAction

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class CustomContextMenu(QtWidgets.QMenu):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.parent = parent
        self.plt_menu = QtWidgets.QMenu("View")
        self.show_plot = ShowPlotAction(parent)
        self.plt_menu.addAction(self.show_plot)
        self.addMenu(self.plt_menu)


class ImageContextMenu(CustomContextMenu):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.addSeparator()
        self.save_slice = QtGui.QAction(
            text="Save slice...",
            parent=parent,
            icon=QtGui.QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "images", "camera.ico"
                ).__str__()
            ),
            # icon=parent.style().standardIcon(
            #     QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            # ),
        )
        self.save_slice.triggered.connect(lambda x: self._save_slice)
        self.addAction(self.save_slice)

    def _save_slice(self):
        """
        Save Slice to PNG Callback.

        The _save_slice function is called when the user clicks on the &quot;Save slice&quot; button.
        It opens a file dialog to allow the user to select where they want to save their image, and then saves it as a PNG
        file.


        Parameters
        ----------
            parent
                Access the parent object, which is the main window

        Returns
        -------
            The file path to the saved image
        """
        if self.parent.data.nii_img.path:
            file_name = self.parent.data.nii_img.path
            file_path = Path(
                QtWidgets.QFileDialog.getSaveFileName(
                    self.parent,
                    caption="Save slice image:",
                    directory=(
                        parent.data.last_dir / (file_name.stem + ".png")
                    ).__str__(),
                    filter="PNG Files (*.png)",
                )[0]
            )
            self.parent.data.last_dir = file_path.parent
        else:
            file_path = None

        if file_path:
            parent.image_axis.figure.savefig(
                file_path, bbox_inches="tight", pad_inches=0
            )
            print("Figure saved:", file_path)


class PlotDecayContextMenu(CustomContextMenu):
    def __init__(self, parent: Main):
        super().__init__(parent)
        self.b_values = np.array([])
        self.load_b_values = QtGui.QAction("Load b-values", self)
        self.load_b_values.setIcon(
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
            )
        )
        self.load_b_values.triggered.connect(self._load_b_values)
        self.addAction(self.load_b_values)

    @property
    def b_values(self):
        return self._b_values

    @b_values.setter
    def b_values(self, value: list | np.ndarray):
        if isinstance(value, list):
            value = np.array(value)
        self.parent.data.fit_data.params.b_values = value
        # if hasattr(self.parent, "plot_layout"):
        self.parent.plot_layout.decay.x_data = value
        self._b_values = value

    def _load_b_values(self):
        """Callback to load b-values from file."""
        path = QtWidgets.QFileDialog.getOpenFileName(
            caption="Open B-Value File",
            directory=self.parent.data.last_dir.__str__(),
        )[0]

        if path:
            file = Path(path)
            with open(file, "r") as f:
                # find away to decide which one is right
                self.b_values = [int(x) for x in f.read().split("\n")]
        else:
            print("No B-Values loaded.")
