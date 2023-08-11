from PyQt6 import QtWidgets, QtCore, QtGui
from pathlib import Path


class SettingsWindow(QtWidgets.QWidget):
    def __init__(
        self, parent: QtWidgets.QMainWindow, settings: QtCore.QSettings
    ) -> None:
        super().__init__()

        self.settings = settings
        self.setWindowTitle("Settings")
        img = Path(Path(__file__).parent, "resources", "Logo.png").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        # self.setWindowIcon(QtGui.QIcon(img))

        # TODO: Would be nice to center -> Use Dlg instead of Widget
        # geometry = self.geometry()
        # geometry.moveCenter(parent.geometry().center())
        # self.setGeometry(geometry)
        # TODO Adjust Size automatically
        self.setMinimumSize(192, 64)

        self.main_layout = QtWidgets.QVBoxLayout()

        general_label = QtWidgets.QLabel("General:")
        self.main_layout.addWidget(general_label)
        general_line = QtWidgets.QFrame()
        general_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        general_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.main_layout.addWidget(general_line)

        self.theme_layout = QtWidgets.QHBoxLayout()
        theme_label = QtWidgets.QLabel("Theme:")
        self.theme_layout.addWidget(theme_label)
        self.theme_combobox = QtWidgets.QComboBox()
        self.theme_combobox.addItems(["Dark", "Light"])
        self.theme_combobox.setCurrentText(settings.value("theme"))
        self.theme_combobox.currentIndexChanged.connect(self._theme_changed)
        self.theme_layout.addWidget(self.theme_combobox)

        self.main_layout.addLayout(self.theme_layout)
        # self.theme_combobox.setItem

        self.setLayout(self.main_layout)

    def _theme_changed(self):
        self.settings.setValue("theme", self.theme_combobox.currentText())
