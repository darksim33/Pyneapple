from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpacerItem,
    QSizePolicy,
)
import importlib.metadata

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class AboutDlg(QDialog):
    """About Dlg displaying Pyneapple package information."""

    main_layout: QHBoxLayout | QHBoxLayout

    def __init__(self, parent: MainWindow = None):
        super().__init__(parent)
        self.parent = parent
        self.metadata = importlib.metadata.metadata("pyneapple")
        self.setup_ui()

    def setup_ui(self):
        """Setup UI elements."""
        self.setWindowTitle("About Pyneapple")
        self.setWindowIcon(
            QIcon(
                (
                    self.parent.data.app_path / "resources" / "images" / "app.ico"
                ).__str__()
            )
        )
        width = 352
        height = 224
        # Fixed Size
        # self.setFixedSize(width, height)
        # self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.setMinimumSize(width, height)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.setMaximumSize(width * 2, height * 2)

        self.main_layout = QHBoxLayout()

        # Icon Display Layout
        icon_layout = QVBoxLayout()
        self.main_layout.addLayout(icon_layout)

        icon_label = QLabel()
        icon_layout.addWidget(icon_label)
        icon_label.setPixmap(
            QPixmap(
                (
                    self.parent.data.app_path / "resources" / "images" / "app.png"
                ).__str__()
            )
        )
        icon_label.setScaledContents(True)
        icon_size = 96
        icon_label.setFixedSize(icon_size, icon_size)

        icon_layout.addSpacerItem(
            QSpacerItem(
                64, 64, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
        )

        # Main Info text field
        text_layout = QVBoxLayout()
        self.main_layout.addLayout(text_layout)
        info_label = QLabel()
        default_font_size = info_label.fontMetrics().size(0, "font").height()
        info_label.setText(
            "<p style='line-height: 120%;'>"
            + f"<strong style='font-size: {default_font_size + 2}px;'>"
            + "Pyneapple </strong> <br> Version: "
            + self.metadata["version"]
            + "<br> Python Version: "
            + self.metadata["Requires-Python"].replace(",<4.0", "")
            + "<br> License: "
            + self.metadata["license"]
            + "<br> Authors: "
            + self.metadata["author"]
            + ", "
            + self.metadata["maintainer"]
            + "<br> Homepage: <a href='"
            + self.metadata["Home-page"]
            + "'>Github/Pyneapple</a>"
            + "<br><br>"
            + "Copyright: Pyneapple Project 2024"
            + "</p>"
        )
        info_label.setOpenExternalLinks(True)
        text_layout.addWidget(info_label)

        # spacer = QSpacerItem(width - icon_size - 6, 4)
        # text_layout.addSpacerItem(spacer)

        text_layout.addStretch()

        self.setLayout(self.main_layout)
