from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpacerItem,
    QSizePolicy,
)
import importlib.metadata

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow

package_data = importlib.metadata.metadata("pyneapple")


class AboutDlg(QDialog):
    """About Dlg displaying Pyneapple package information."""

    text_label: QLabel
    icon_label: QLabel
    main_layout: QHBoxLayout

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.metadata = importlib.metadata.metadata("pyneapple")
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("About Pyneapple")
        self.setWindowIcon(
            QIcon(
                (
                    self.parent.data.app_path / "resources" / "images" / "app.ico"
                ).__str__()
            )
        )
        # self.setMinimumSize(500, 500)
        width = 352
        self.setFixedSize(width, 256)

        self.main_layout = QHBoxLayout()
        icon_layout = QVBoxLayout()

        self.icon_label = QLabel()
        self.icon_label.setPixmap(
            QPixmap(
                (
                    self.parent.data.app_path / "resources" / "images" / "app.png"
                ).__str__()
            )
        )
        self.icon_label.setScaledContents(True)
        icon_size = 96
        # self.icon_label.setMinimumSize(icon_size, icon_size)
        # self.icon_label.setMaximumSize(icon_size, icon_size)
        self.icon_label.setFixedSize(icon_size, icon_size)

        icon_layout.addWidget(self.icon_label)
        icon_layout.addSpacerItem(
            QSpacerItem(
                64, 64, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
        )

        self.main_layout.addLayout(icon_layout)

        text_layout = QVBoxLayout()
        title = QLabel()
        title.setText(f"Pyneapple")
        text_layout.addWidget(title)
        version = QLabel()
        version.setText(f"Version: " + self.metadata["version"])
        text_layout.addWidget(version)

        spacer = QLabel()
        spacer.setMinimumWidth(width - icon_size - 10)
        text_layout.addWidget(spacer)

        # text_layout.addSpacerItem(
        #     QSpacerItem(
        #         64, 64, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        #     )
        # )

        text_layout.addStretch()
        self.main_layout.addLayout(text_layout)

        self.setLayout(self.main_layout)
