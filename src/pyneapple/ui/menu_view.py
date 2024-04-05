from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod
from pathlib import Path

from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QIcon

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class SwitchImageAction(QAction):
    def __init__(self, parent: MainWindow, text: str):
        """Basic class to handle image switching (abstract)."""
        super().__init__(parent=parent, text=text)
        self.parent = parent
        self.triggered.connect(self.switch)

    @abstractmethod
    def switch(self):
        pass


class SwitchToSegmentedImageAction(SwitchImageAction):
    def __init__(self, parent: MainWindow):
        """Action to switch to segmented image."""
        super().__init__(parent=parent, text="Image with applied Mask")
        self.img_type = "Img"
        self.setEnabled(False)

    def switch(self):
        """Switch Image Callback"""
        self.parent.settings.setValue("img_disp_type", self.img_type)
        self.parent.image_axis.setup_image()


class ShowPlotAction(QAction):
    def __init__(self, parent: MainWindow):
        """Action that handles toggling of the plots canvas on the right."""
        super().__init__(
            parent=parent,
            text="Show Plot",
            icon=QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "images", "graph.ico"
                ).__str__()
            ),
        )
        self.parent = parent
        self.setEnabled(True)
        self.setCheckable(True)
        self.triggered.connect(self.show)

    def show(self):
        """Plot Axis show Callback"""
        if not self.isChecked():
            self.parent.main_hLayout.removeItem(self.parent.plot_layout)
            self.parent.settings.setValue("plt_show", True)
        else:
            self.parent.main_hLayout.addLayout(self.parent.plot_layout)
            self.parent.settings.setValue("plt_show", True)
        self.parent.image_axis.resize_figure_axis()
        self.parent.image_axis.resize_canvas()
        self.parent.image_axis.setup_image()


class PlotDisplayTypeSingleVoxelAction(QAction):
    def __init__(self, parent: MainWindow):
        """Action to set the plot axis to display the data of each voxel."""
        super().__init__(parent=parent, text="Show Single Voxel Spectrum")
        self.parent = parent
        self.setCheckable(True)
        self.setChecked(True)
        self.setEnabled(False)


class PlotDisplayTypeSegmentationAction(QAction):
    def __init__(self, parent: MainWindow):
        """Action to set the plot axis to display the data of each segmentation."""
        super().__init__(parent=parent, text="Show Segmentation Spectrum")
        self.parent = parent
        self.setCheckable(True)
        self.setChecked(False)
        self.setEnabled(False)


class ShowSegmentationOverlayAction(QAction):
    def __init__(self, parent: MainWindow):
        """Action to show the segmentation as overlay on the main image canvas."""
        super().__init__(parent=parent, text="Show Segmentation Overlay")
        self.parent = parent
        self.triggered.connect(self.show)

    def show(self):
        """Show the segmentation as overlay on the main image canvas."""
        self.parent.settings.setValue(
            "img_disp_overlay", True if self.isChecked() else False
        )
        self.parent.image_axis.setup_image()


class ViewMenu(QMenu):
    switch2segmented: SwitchToSegmentedImageAction
    plt_show: ShowPlotAction
    plt_display_type_single_voxel: PlotDisplayTypeSingleVoxelAction
    plt_display_type_segmentation: PlotDisplayTypeSegmentationAction
    show_seg_overlay: ShowSegmentationOverlayAction

    def __init__(self, parent: MainWindow):
        """
        QMenu to handle the basic viewing related actions.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the menu
        """
        super().__init__("&View", parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Sets up menu."""
        switch_image_menu = QMenu("Switch Image", parent=self)
        self.switch2segmented = SwitchToSegmentedImageAction(self.parent)
        switch_image_menu.addAction(self.switch2segmented)
        # self.addMenu(switch_image_menu)

        self.plt_show = ShowPlotAction(self.parent)
        self.addAction(self.plt_show)
        # self.addSeparator()

        self.plt_display_type_single_voxel = PlotDisplayTypeSingleVoxelAction(
            self.parent
        )
        # self.addAction(self.plt_display_type_single_voxel)
        self.plt_display_type_segmentation = PlotDisplayTypeSegmentationAction(
            self.parent
        )
        # self.addAction(self.plt_display_type_segmentation)
        # self.addSeparator()

        self.show_seg_overlay = ShowSegmentationOverlayAction(self.parent)
        # self.addAction(self.show_seg_overlay)
