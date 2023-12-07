from PyQt6 import QtWidgets, QtCore
from PIL import Image
from pathlib import Path
from copy import deepcopy

from PyQt6.QtWidgets import QSpinBox, QVBoxLayout, QHBoxLayout, QSlider
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    FigureCanvasQTAgg,
)
import matplotlib.patches as patches
from matplotlib.figure import Figure

from src.utils import Nii, NiiSeg


class ImageCanvas(QtWidgets.QVBoxLayout):
    figure: Figure
    canvas: FigureCanvasQTAgg
    axis: Axes
    slider: QSlider
    spinbox: QSpinBox
    main_layout: QVBoxLayout
    slice_layout: QHBoxLayout

    def __init__(
        self,
        image: Nii = Nii(),
        segmentation: NiiSeg = NiiSeg(),
        settings: dict = None,  # settings is the appdata plt settings dict
        window_width: int = 0,
        theme: str = "Light",
    ):
        super().__init__()

        self.setup_ui()

        self._image = image
        self._segmentation = segmentation
        self.settings = settings
        self._theme = theme
        self.window_width = window_width

        self.deploy_default_image()
        self.resize_canvas()
        self.resize_figure_axis()

    def setup_ui(self):
        # init basic figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
        )
        self.axis = self.figure.add_subplot(111)
        self.addWidget(self.canvas)
        self.axis.clear()

        # Slice Layout
        self.slice_layout = QtWidgets.QHBoxLayout()

        # Slider
        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.valueChanged.connect(lambda x: self._slice_slider_changed())
        self.slice_layout.addWidget(self.slider)

        # SpinBox
        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setValue(1)
        self.spinbox.setEnabled(False)
        self.spinbox.setMinimumWidth(20)
        self.spinbox.setMaximumWidth(40)
        self.spinbox.valueChanged.connect(lambda x: self._slice_spn_bx_changed())
        self.slice_layout.addWidget(self.spinbox)

        self.addLayout(self.slice_layout)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, nii: Nii):
        if nii.path:
            self._image = nii
            self.setup_image()

            self.slider.setEnabled(True)
            self.slider.setMaximum(nii.array.shape[2])
            self.spinbox.setEnabled(True)
            self.spinbox.setMaximum(nii.array.shape[2])

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
                    Path(
                        Path(__file__).parent.parent.parent,
                        "resources",
                        "PyNeappleLogo_gray.png",
                    ).__str__()
                ),
                cmap="gray",
            )
            self.figure.set_facecolor("black")
        elif self._theme == "Light":
            self.axis.imshow(
                Image.open(
                    Path(
                        Path(__file__).parent.parent.parent,
                        "resources",
                        "PyNeappleLogo_gray_text.png",
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
            self.slider.setMaximumWidth(
                round(self.window_width * 0.6) - self.spinbox.width()
            )
        else:
            self.canvas.setMaximumWidth(16777215)
            self.slider.setMaximumWidth(16777215)
        # FIXME After deactivating the Plot the Canvas expands but wont fill the whole window

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
        self.settings["n_slice"].number = self.spinbox.value()

        if self.image.path:
            img_display = self.image.to_rgba_array(self.settings["n_slice"].value)
            self.axis.clear()
            self.axis.imshow(img_display, cmap="gray")
            if self.settings["show_segmentation"] and self.segmentation.path:
                self.deploy_segmentation()

            self.axis.axis("off")
            self.resize_canvas()
            self.resize_figure_axis()
            self.canvas.draw()

    def deploy_segmentation(self):
        """Draw segmentation on Canvas"""
        # get colors
        colors = self.settings["seg_colors"]
        if self.segmentation.segmentations:
            seg_color_idx = 0
            for seg_number in self.segmentation.segmentations:
                segmentation = self.segmentation.segmentations[seg_number]
                if self.settings["n_slice"].value in segmentation.polygon_patches:
                    polygon_patch: patches.Polygon
                    for polygon_patch in segmentation.polygon_patches[
                        self.settings["n_slice"].value
                    ]:
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

    def _slice_slider_changed(self):
        """Slice Slider Callback"""
        self.settings["n_slice"].number = self.slider.value()
        self.spinbox.setValue(self.slider.value())
        self.setup_image()

    def _slice_spn_bx_changed(self):
        """Slice Spinbox Callback"""
        self.settings["n_slice"].number = self.spinbox.value()
        self.slider.setValue(self.spinbox.value())
        self.setup_image()

    def clear(self):
        """Clear Widget"""
        # Reset UI
        self.slider.setEnabled(False)
        self.slider.setValue(0)
        self.spinbox.setEnabled(False)
        self.spinbox.setValue(0)

        # Clear Image
        self.axis.clear()
        self.deploy_default_image()

        # Remove data from Widget
        self.image = Nii()
        self.segmentation = NiiSeg()
