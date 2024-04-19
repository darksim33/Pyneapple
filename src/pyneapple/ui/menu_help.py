from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QIcon

from .dlg_about import AboutDlg

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class AboutAction(QAction):
    def __init__(self, parent: MainWindow) -> None:
        """
        QAction for QMenu to display an About Dialog
        Parameters
        ----------
        parent: MainWindow
        """
        super().__init__(parent=parent, text="About")
        self.parent = parent
        self.setIcon(
            QIcon(
                (
                    self.parent.data.app_path / "resources" / "images" / "app.ico"
                ).__str__()
            )
        )
        self.triggered.connect(self.open_about)

    def open_about(self) -> None:
        dlg = AboutDlg(self.parent)
        dlg.exec()


class HelpMenu(QMenu):
    about: AboutAction

    def __init__(self, parent: MainWindow = None):
        """
        QMenu handling the help menu.

        Parameters:
            parent:
                Pass the parent window to the menu
        """
        super().__init__("&Help", parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI elements."""
        self.about = AboutAction(self.parent)
        self.addAction(self.about)
