from __future__ import annotations

from src.ui.menubar.file_menu import FileMenu
from src.ui.menubar.edit_menu import EditMenu
from src.ui.menubar.fitting_menu import FittingMenu
from src.ui.menubar.view_menu import ViewMenu

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyNeapple_UI import MainWindow


class MenuBarBuilder:
    def __init__(self, parent: MainWindow):
        self.parent = parent
        self.setup_menubar()

    def setup_menubar(self):
        menubar = self.parent.menuBar()
        self.parent.file_menu = FileMenu(self.parent)
        menubar.addMenu(self.parent.file_menu)
        self.parent.edit_menu = EditMenu(self.parent)
        menubar.addMenu(self.parent.edit_menu)
        self.parent.fitting_menu = FittingMenu(self.parent)
        menubar.addMenu(self.parent.fitting_menu)
        self.parent.view_menu = ViewMenu(self.parent)
        menubar.addMenu(self.parent.view_menu)
