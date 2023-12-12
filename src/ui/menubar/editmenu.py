from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QIcon

from src.utils import Processing

if TYPE_CHECKING:
    from PyneappleUI import MainWindow


class ImageZeroPadding(QAction):
    def __init__(self, parent: MainWindow):
        super().__init__(text="For Image", parent=parent)
        self.triggered.connect(parent.data.nii_img.zero_padding)


class SegmentationZeroPadding(QAction):
    def __init__(self, parent: MainWindow):
        super().__init__(text="For Segmentation", parent=parent)
        # self.pad_seg.triggerd.connect(self.data.nii_seg.super().zero_padding)


class RotMaskAction(QAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            text="&Rotate Mask clockwise",
            parent=parent,
            icon=QIcon(
                Path(Path(parent.data.app_path), "resources", "rot90.png").__str__()
            ),
        )
        self.parent = parent
        self.setEnabled(False)
        self.triggered.connect(self.rotate)

    def rotate(self):
        pass


class FlipAction(QAction):
    def __init__(self, parent: MainWindow, text: str, icon: QIcon):
        super().__init__(text=text, parent=parent, icon=icon)
        self.parent = parent
        self.setEnabled(False)
        self.triggered.connect(self.flip)

    @abstractmethod
    def flip(self):
        pass


class FlipMaskUpDownAction(FlipAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            text="Flip Mask Up-Down",
            parent=parent,
            icon=QIcon(
                Path(Path(parent.data.app_path), "resources", "flipUD.png").__str__()
            ),
        )

    def flip(self):
        """Flip mask up and down. (Superior/inferior)"""
        self.parent.data.nii_seg.array = np.fliplr(self.parent.data.nii_seg.array)
        self.parent.data.nii_seg.calculate_polygons()
        self.parent.image_axis.setup_image()


class FlipMaskBackForthAction(FlipAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            text="Flip Mask Back-Forth",
            parent=parent,
            icon=QIcon(
                Path(Path(parent.data.app_path), "resources", "flipZ.png").__str__()
            ),
        )

    def flip(self):
        self.parent.data.nii_seg.array = np.flip(self.parent.data.nii_seg.array, axis=2)
        self.parent.data.nii_seg.calculate_polygons()
        self.parent.image_axis.setup_image()


class FlipMaskLeftRightAction(FlipAction):
    def __init__(self, parent: MainWindow):
        super().__init__(
            text="Flip Mask Left-Right",
            parent=parent,
            icon=QIcon(
                Path(Path(parent.data.app_path), "resources", "flipLR.png").__str__()
            ),
        )

    def flip(self):
        self.parent.data.nii_seg.array = np.flipud(self.parent.data.nii_seg.array)
        self.parent.data.nii_seg.calculate_polygons()
        self.parent.image_axis.setup_image()


class SegmentationToImageAction(QAction):
    def __init__(self, parent: MainWindow):
        super().__init__(text="&Apply on Image", parent=parent)
        self.parent = parent
        self.triggered.connect(self.seg2img)

    def seg2img(self):
        self.parent.data.nii_img_masked = Processing.merge_nii_images(
            self.parent.data.nii_img, self.parent.data.nii_seg
        )
        if self.parent.data.nii_img_masked:
            self.parent.plt_showMaskedImage.setEnabled(True)
            self.parent.saveMaskedImage.setEnabled(True)


class EditMenu(QMenu):
    img_padding: ImageZeroPadding
    seg_padding: SegmentationZeroPadding
    mask_rotate: RotMaskAction
    mask_flip_up_down: FlipMaskUpDownAction
    mask_flip_left_right: FlipMaskLeftRightAction
    mask_flip_back_forth: FlipMaskBackForthAction
    seg2img: SegmentationToImageAction

    def __init__(self, parent: MainWindow):
        super().__init__("&Edit", parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        # Padding
        padding_menu = QMenu("Zero-Padding", self.parent)
        self.img_padding = ImageZeroPadding(self.parent)
        padding_menu.addAction(self.img_padding)
        self.seg_padding = SegmentationZeroPadding(self.parent)
        padding_menu.addAction(self.seg_padding)
        self.addMenu(padding_menu)

        # Mask Tools
        mask_menu = QMenu("&Mask Tools", self.parent)

        orientation_menu = QMenu("&Orientation", self.parent)
        self.mask_rotate = RotMaskAction(self.parent)
        orientation_menu.addAction(self.mask_rotate)
        self.mask_flip_up_down = FlipMaskUpDownAction(self.parent)
        orientation_menu.addAction(self.mask_flip_up_down)
        self.mask_flip_left_right = FlipMaskLeftRightAction(self.parent)
        orientation_menu.addAction(self.mask_flip_left_right)
        self.mask_flip_back_forth = FlipMaskBackForthAction(self.parent)
        orientation_menu.addAction(self.mask_flip_back_forth)
        mask_menu.addMenu(orientation_menu)

        self.seg2img = SegmentationToImageAction(self.parent)
        mask_menu.addAction(self.seg2img)
        self.addMenu(mask_menu)
