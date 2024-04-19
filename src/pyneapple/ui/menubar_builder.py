from __future__ import annotations
from .menu_file import FileMenu
from .menu_edit import EditMenu
from .menu_fitting import FittingMenu
from .menu_help import HelpMenu
from .menu_view import ViewMenu

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class MenubarBuilder:
    def __init__(self, parent: MainWindow):
        """
        Menubar building class.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the menu
        """
        self.parent = parent
        self.setup_menubar()

    def setup_menubar(self):
        """Sets up menubar."""
        menubar = self.parent.menuBar()
        self.parent.file_menu = FileMenu(self.parent)
        menubar.addMenu(self.parent.file_menu)
        self.parent.edit_menu = EditMenu(self.parent)
        menubar.addMenu(self.parent.edit_menu)
        self.parent.fitting_menu = FittingMenu(self.parent)
        menubar.addMenu(self.parent.fitting_menu)
        self.parent.view_menu = ViewMenu(self.parent)
        menubar.addMenu(self.parent.view_menu)
        self.parent.help_menu = HelpMenu(self.parent)
        menubar.addMenu(self.parent.help_menu)
