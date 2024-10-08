from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QIcon

from pyneapple.utils import processing
from .dlg_prompts import ZeroPaddingMissmatchMessageBox

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class ImageZeroPadding(QAction):
    def __init__(self, parent: MainWindow):
        """Image zero-padding action."""
        super().__init__(text="For Image", parent=parent)
        self.parent = parent
        self.triggered.connect(self.pad_image)

    def pad_image(self):
        if not (
            self.parent.data.nii_img.array.shape[0]
            == self.parent.data.nii_img.array.shape[1]
        ):
            if self.parent.data.nii_seg.path:
                if (
                    self.parent.data.nii_img.array.shape[0:2]
                    == self.parent.data.nii_seg.array.shape[0:2]
                ):
                    dlg = ZeroPaddingMissmatchMessageBox()
                    if dlg.exec() == ZeroPaddingMissmatchMessageBox.StandardButton.Yes:
                        self.parent.data.nii_seg.zero_padding()
                    self.parent.data.nii_img.zero_padding()
                    print(
                        f"Padded Image to {self.parent.data.nii_img.array.shape[0]},{self.parent.data.nii_img.array.shape[1]}"
                    )
                    self.parent.image_axis.setup_image()


class SegmentationZeroPadding(QAction):
    def __init__(self, parent: MainWindow):
        """Segmentation zero-padding action."""
        super().__init__(text="For Segmentation", parent=parent)
        self.parent = parent
        self.triggered.connect(self.pad_img)
        # self.pad_seg.triggerd.connect(self.data.nii_seg.super().zero_padding)

    def pad_img(self):
        if not (
            self.parent.data.nii_seg.array.shape[0]
            == self.parent.data.nii_seg.array.shape[1]
        ):
            if self.parent.data.nii_img.path:
                if (
                    ZeroPaddingMissmatchMessageBox().exec()
                    == ZeroPaddingMissmatchMessageBox.StandardButton.Yes
                ):
                    self.parent.data.nii_img.zero_padding()
                self.parent.data.nii_seg.zero_padding()
                print(
                    f"Padded Image to {self.parent.data.nii_img.array.shape[0]},{self.parent.data.nii_img.array.shape[1]}"
                )
                self.parent.image_axis.setup_image()


class RotSegmentationAction(QAction):
    def __init__(self, parent: MainWindow):
        """Action to rotate the segmentation clockwise."""
        super().__init__(
            text="&Rotate Mask clockwise",
            parent=parent,
            icon=QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "images", "rot_90.ico"
                ).__str__()
            ),
        )
        self.parent = parent
        self.setEnabled(False)
        self.triggered.connect(self.rotate)

    def rotate(self):
        """Rotate segmentation clockwise."""
        pass


class FlipAction(QAction):
    def __init__(self, parent: MainWindow, text: str, icon: QIcon):
        """Basic flipping class (abstract)."""
        super().__init__(text=text, parent=parent, icon=icon)
        self.parent = parent
        self.setEnabled(False)
        self.triggered.connect(self.flip)

    @abstractmethod
    def flip(self):
        """
        Handle segmentation flip. Needs to be deployed.
        """
        pass


class FlipMaskUpDownAction(FlipAction):
    def __init__(self, parent: MainWindow):
        """Flip segmentation up/down Action."""
        super().__init__(
            text="Flip Mask Up-Down",
            parent=parent,
            icon=QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "images", "flip_ud.ico"
                ).__str__()
            ),
        )

    def flip(self):
        """
        Flip segmentation up/down. (Superior/inferior)

        Since the displayed directions are orientated on the physiological directions the matrix is rotated.
        """
        self.parent.data.nii_seg.array = np.fliplr(self.parent.data.nii_seg.array)
        self.parent.data.nii_seg.calculate_polygons()
        self.parent.image_axis.setup_image()


class FlipMaskBackForthAction(FlipAction):
    def __init__(self, parent: MainWindow):
        """Flip segmentation back/forth Action."""
        super().__init__(
            text="Flip Mask Back-Forth",
            parent=parent,
            icon=QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "images", "flip_bf.ico"
                ).__str__()
            ),
        )

    def flip(self):
        """
        Flip segmentation back/forth

        Since the displayed directions are orientated on the physiological directions the matrix is rotated.
        """
        self.parent.data.nii_seg.array = np.flip(self.parent.data.nii_seg.array, axis=2)
        self.parent.data.nii_seg.calculate_polygons()
        self.parent.image_axis.setup_image()


class FlipMaskLeftRightAction(FlipAction):
    def __init__(self, parent: MainWindow):
        """Flip segmentation left/right Action."""
        super().__init__(
            text="Flip Mask Left-Right",
            parent=parent,
            icon=QIcon(
                Path(
                    Path(parent.data.app_path), "resources", "images", "flip_lr.ico"
                ).__str__()
            ),
        )

    def flip(self):
        """
        Flip segmentation left/right.

        Since the displayed directions are orientated on the physiological directions the matrix is rotated.
        """
        self.parent.data.nii_seg.array = np.flipud(self.parent.data.nii_seg.array)
        self.parent.data.nii_seg.calculate_polygons()
        self.parent.image_axis.setup_image()


class SegmentationToImageAction(QAction):
    def __init__(self, parent: MainWindow):
        """Action to apply segmentation to image."""
        super().__init__(text="&Apply on Image", parent=parent)
        self.parent = parent
        self.triggered.connect(self.seg2img)

    def seg2img(self):
        """Apply the current segmentation to the loaded image."""
        self.parent.data.nii_img_masked = processing.merge_nii_images(
            self.parent.data.nii_img, self.parent.data.nii_seg
        )
        if self.parent.data.nii_img_masked:
            self.parent.view_menu.switch2segmented.setEnabled(True)
            self.parent.file_menu.save_segmented_image.setEnabled(True)


class EditMenu(QMenu):
    img_padding: ImageZeroPadding
    seg_padding: SegmentationZeroPadding
    seg_rotate: RotSegmentationAction
    seg_flip_up_down: FlipMaskUpDownAction
    seg_flip_left_right: FlipMaskLeftRightAction
    seg_flip_back_forth: FlipMaskBackForthAction
    seg2img: SegmentationToImageAction

    def __init__(self, parent: MainWindow):
        """
        QMenu to handle the basic editing related actions.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the menu
        """
        super().__init__("&Edit", parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Sets up menu."""
        # Padding
        padding_menu = QMenu("Zero-Padding", parent=self)
        self.img_padding = ImageZeroPadding(self.parent)
        padding_menu.addAction(self.img_padding)
        self.seg_padding = SegmentationZeroPadding(self.parent)
        padding_menu.addAction(self.seg_padding)
        self.addMenu(padding_menu)

        # Mask Tools
        mask_menu = QMenu("&Mask Tools", parent=self)

        orientation_menu = QMenu("&Orientation", parent=mask_menu)
        self.seg_rotate = RotSegmentationAction(self.parent)
        orientation_menu.addAction(self.seg_rotate)
        self.seg_flip_up_down = FlipMaskUpDownAction(self.parent)
        orientation_menu.addAction(self.seg_flip_up_down)
        self.seg_flip_left_right = FlipMaskLeftRightAction(self.parent)
        orientation_menu.addAction(self.seg_flip_left_right)
        self.seg_flip_back_forth = FlipMaskBackForthAction(self.parent)
        orientation_menu.addAction(self.seg_flip_back_forth)
        mask_menu.addMenu(orientation_menu)

        self.seg2img = SegmentationToImageAction(self.parent)
        mask_menu.addAction(self.seg2img)
        self.addMenu(mask_menu)
