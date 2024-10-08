from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6 import QtWidgets, QtCore
from PIL import Image
from pathlib import Path
from copy import deepcopy

from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QScrollBar,
    QLabel,
    QLineEdit,
)
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    FigureCanvasQTAgg,
)
import matplotlib.patches as patches
from matplotlib.figure import Figure

from nifti import Nii, NiiSeg

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class SliceNumberEdit(QLineEdit):
    _value: int | str

    def __init__(self, value: int | float | str = None):
        """
        Displays slice number and allows edits.

        Works like a classic pyqt widget with non property setter getter behaviour.
        """
        super().__init__()
        self.min = 1  # in the context of slices these are "numbers" not value.
        self.max = 1
        self.setValue(value)
        self.textChanged.connect(self.value_changed)
        self.setValidator(QIntValidator(1, 100000))
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

    # PyQt Style Getter Setter behaviour
    def value(self):
        return int(self._value)

    def setValue(self, value: int | float | str | None):
        # for SpinBox like Qt setting
        if (
            not isinstance(value, (int, float))
            or not isinstance(value, str)
            or not None
        ):
            pass
        else:
            raise TypeError(f"Value needs to be either integer, float or string.")

        if value is not None:
            try:
                value = int(value)
                if value < self.min:
                    value = self.min
                elif value > self.max:
                    value = self.max
            except TypeError:
                raise TypeError()
        else:
            value = 1

        self.setText(str(value))
        self.setToolTip(f"Slice {value} of {self.max}")
        self._value = value

    def setMaximum(self, value):
        tooltip = f"(Slice {self.value()} of {value})"
        self.setToolTip(tooltip)
        self.max = value

    def setMinimum(self, value):
        self.min = value

    def value_changed(self):
        text = self.text()
        if not text == "" and text.isdigit():
            self.setValue(int(text))


class ImageCanvas(QtWidgets.QGridLayout):
    pos_label: QLabel | QLabel
    figure: Figure
    canvas: FigureCanvasQTAgg
    axis: Axes
    scrollbar: QScrollBar
    slice_number_edit: SliceNumberEdit
    main_layout: QVBoxLayout
    canvas_layout: QHBoxLayout

    def __init__(
        self,
        parent: MainWindow,
        image: Nii = Nii(),
        segmentation: NiiSeg = NiiSeg(),
        settings: dict = None,  # settings is the appdata plt settings dict
        window_width: int = 0,
        theme: str = "Light",
    ):
        super().__init__()

        self._image = image
        self._segmentation = segmentation
        self.parent = parent
        self.settings = settings
        self._theme = theme
        self.window_width = window_width
        self.scrollbar = QScrollBar()
        self.slice_number_edit = SliceNumberEdit(1)

        self.slice_number = 1
        self.setup_ui()

        self.deploy_default_image()
        self.resize_canvas()
        self.resize_figure_axis()

    # The two properties slice_number and slice_value are used to connect array indexes to slice numbers

    @property
    def slice_number(self):
        return self._slice_value + 1

    @slice_number.setter
    def slice_number(self, value):
        # check if the set number is in the range of allowed values (min & max are treated like "numbers")
        if self.slice_number_edit.min > value:
            value = self.slice_number_edit.min
        elif self.slice_number_edit.max < value:
            value = self.slice_number_edit.max

        self.settings["n_slice"].number = value
        self._slice_value = value - 1
        # Check if Scrollbar and EditField are in sync
        if not value == self.scrollbar.value():
            self.scrollbar.setValue(value)
        if not value == self.slice_number_edit.value():
            self.slice_number_edit.setValue(value)

    @property
    def slice_value(self):
        return self._slice_value

    @slice_value.setter
    def slice_value(self, value):
        if self.slice_number_edit.min > value + 1:
            value = self.slice_number_edit.min
        elif self.slice_number_edit.max < value + 1:
            value = self.slice_number_edit.max

        self.settings["n_slice"].value = value
        self._slice_value = value
        # Check if Scrollbar and EditField are in sync
        if not value + 1 == self.scrollbar.value():
            self.scrollbar.setValue(value + 1)
        if not value + 1 == self.slice_number_edit.value():
            self.slice_number_edit.setValue(value + 1)

    def setup_ui(self):
        # Canvas Layout
        self.canvas_layout = QtWidgets.QHBoxLayout()

        # init basic figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
        )
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.axis = self.figure.add_subplot(111)
        self.canvas_layout.addWidget(self.canvas)
        self.axis.clear()

        # Scrollbar
        self.scrollbar = QtWidgets.QScrollBar()
        self.scrollbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.scrollbar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )
        self.scrollbar.setFocusPolicy(QtCore.Qt.FocusPolicy.WheelFocus)
        self.scrollbar.setPageStep(self.slice_number)
        self.scrollbar.setEnabled(False)
        self.scrollbar.setMinimum(1)
        self.scrollbar.setMaximum(1)
        self.scrollbar.valueChanged.connect(lambda x: self._scrollbar_changed())
        self.canvas_layout.addWidget(self.scrollbar)

        self.addLayout(self.canvas_layout, 0, 0)

        # Slice info layout
        slice_layout = QHBoxLayout()

        self.pos_label = QLabel("")
        self.pos_label.setStyleSheet("QLabel {color: gray}")
        slice_layout.addWidget(self.pos_label)

        self.slice_number_edit = SliceNumberEdit()
        self.slice_number_edit.setValue(1)
        self.slice_number_edit.setEnabled(False)
        self.slice_number_edit.setMinimumWidth(20)
        self.slice_number_edit.setMaximumWidth(30)
        self.slice_number_edit.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.slice_number_edit.setFocus(QtCore.Qt.FocusReason.NoFocusReason)
        self.slice_number_edit.textChanged.connect(
            lambda x: self._slice_number_changed()
        )
        self.slice_number_edit.setToolTip(
            f"Slice {self.slice_number_edit.text()} of {self.slice_number_edit.text()}"
        )
        slice_layout.addWidget(self.slice_number_edit)
        self.addLayout(slice_layout, 1, 0)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, nii: Nii):
        if nii.path:
            self._image = nii
            self.setup_image()

            self.scrollbar.setEnabled(True)
            self.scrollbar.setMaximum(nii.array.shape[2])
            self.slice_number_edit.setEnabled(True)
            self.slice_number_edit.setMaximum(nii.array.shape[2])

    @property
    def segmentation(self):
        return self._segmentation

    @segmentation.setter
    def segmentation(self, nii: NiiSeg):
        if nii.path:
            self._segmentation = nii
            if self.settings["show_segmentation"]:
                self.deploy_segmentation()
                self.canvas.draw()

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, theme: str):
        self._theme = theme
        if self.image.path is None:
            self.deploy_default_image()

    def deploy_event(self, handle_name, event):
        """Connect Events to Canvas"""
        self.canvas.mpl_connect(handle_name, event)

    def deploy_default_image(self):
        """Set default Image"""
        self.axis.clear()
        if self._theme == "Dark" or self._theme == "Fusion":
            self.axis.imshow(
                Image.open(
                    (
                        self.parent.data.app_path
                        / "resources"
                        / "images"
                        / "background_gray.png"
                    )
                ),
                cmap="gray",
            )
            self.figure.set_facecolor((0.1, 0.1, 0.1))
        elif self._theme == "Light":
            self.axis.imshow(
                Image.open(
                    (
                        self.parent.data.app_path
                        / "resources"
                        / "images"
                        / "background_gray.png"
                    )
                ),
                cmap="gray",
            )
            self.figure.set_facecolor("white")
        self.axis.axis("off")
        self.resize_canvas()
        self.resize_figure_axis()
        self.canvas.draw()

    def resize_canvas(self):
        if self.settings["show_plot"]:
            # Canvas size should not exceed 60% of the main windows size so that the graphs can be displayed properly
            self.canvas.setMaximumWidth(round(self.window_width * 0.6))
            self.scrollbar.setMaximumWidth(
                round(self.window_width * 0.6) - self.slice_number_edit.width()
            )
        else:
            self.canvas.setMaximumWidth(16777215)
            self.scrollbar.setMaximumWidth(16777215)

    def resize_figure_axis(self, aspect_ratio: tuple | None = (1.0, 1.0)):
        """Resize main image axis to canvas size"""
        box = self.axis.get_position()
        if box.width > box.height:
            # fix height
            scaling = aspect_ratio[0] / box.width
            new_height = box.height * scaling
            new_y0 = (1 - new_height) / 2
            self.axis.set_position(
                [(1 - aspect_ratio[0]) / 2, new_y0, aspect_ratio[1], new_height]
            )
        elif box.width < box.height:
            # fix width
            scaling = aspect_ratio[1] / box.height
            new_width = box.width * scaling
            new_x0 = (1 - new_width) / 2
            self.axis.set_position(
                [new_x0, (1 - aspect_ratio[0]) / 2, new_width, aspect_ratio[1]]
            )

    def setup_image(self):
        """Setup image on canvas"""

        # get current Slice
        # self.slice_number = self.slice_number_edit.value()

        if self.image.path:
            img_display = self.image.to_rgba_array(self.slice_value)
            self.axis.clear()
            self.axis.imshow(img_display, cmap="gray")
            if self.settings["show_segmentation"] and self.segmentation.path:
                self.deploy_segmentation()

            self.axis.axis("off")
            self.resize_canvas()
            self.resize_figure_axis()
            self.canvas.draw()
            self.slice_number_edit.setFocus(QtCore.Qt.FocusReason.NoFocusReason)

    def deploy_segmentation(self):
        """Draw segmentation on Canvas"""
        # get colors
        colors = self.settings["seg_colors"]
        if self.segmentation.segmentations:
            seg_color_idx = 0
            for seg_number in self.segmentation.segmentations:
                segmentation = self.segmentation.segmentations[seg_number]
                if self.slice_value in segmentation.polygon_patches:
                    polygon_patch: patches.Polygon
                    for polygon_patch in segmentation.polygon_patches[self.slice_value]:
                        if not colors[seg_color_idx] == "None":
                            # Two Polygons are drawn to set different alpha for the edge and the face
                            # Setup Face (below edge)
                            polygon_patch_face = deepcopy(polygon_patch)
                            polygon_patch_face.set_facecolor(colors[seg_color_idx])
                            polygon_patch_face.set_alpha(
                                self.settings["seg_face_alpha"]
                            )
                            polygon_patch_face.set_edgecolor("none")
                            self.axis.add_patch(polygon_patch_face)

                            # Setup Edge
                            polygon_path_edge = deepcopy(polygon_patch)
                            polygon_path_edge.set_edgecolor(colors[seg_color_idx])
                            polygon_path_edge.set_alpha(self.settings["seg_edge_alpha"])
                            polygon_path_edge.set_linewidth(
                                self.settings["seg_line_width"]
                            )
                            polygon_path_edge.set_facecolor("none")
                            self.axis.add_patch(polygon_path_edge)

                seg_color_idx += 1

    def _scrollbar_changed(self):
        """Slice Slider Callback"""
        self.slice_number = self.scrollbar.value()
        # self.slice_number_edit.setValue(self.scrollbar.value())
        self.setup_image()

    def _slice_number_changed(self):
        """Slice Number Callback"""
        self.slice_number_edit.value_changed()
        self.slice_number = self.slice_number_edit.value()
        # self.scrollbar.setValue(self.slice_number_edit.value())
        self.setup_image()

    def clear(self):
        """Clear Widget"""
        # Reset UI
        self.slice_number = 1
        self.scrollbar.setEnabled(False)
        self.slice_number_edit.setEnabled(False)

        # Clear Image
        self.axis.clear()
        self.deploy_default_image()

        # Remove data from Widget
        self.image.clear()
        self.segmentation.clear()

    def on_scroll(self, event):
        increment = -1 if event.button == "up" else 1
        self.slice_number = self.slice_number + increment
        self.setup_image()

    def on_click(self, event):
        # On Click events are handled by the "event filter" and deployed in the main pyneapple gui
        pass
