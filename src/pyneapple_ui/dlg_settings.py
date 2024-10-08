from __future__ import annotations
from PyQt6 import QtWidgets, QtCore, QtGui
from pathlib import Path
from os import cpu_count

from .appdata import AppData


class TopicSeperator(QtWidgets.QVBoxLayout):
    """Settings topics Seperator Layout with Label"""

    def __init__(self, name: str, text: str):
        super().__init__()
        self.name = name
        general_label = QtWidgets.QLabel(text)
        self.addWidget(general_label)
        general_line = QtWidgets.QFrame()
        general_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        general_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.addWidget(general_line)


class BasicHLayout(QtWidgets.QHBoxLayout):
    """
    Basic horizontal layout for settings dlg

    Attributes:
    ----------
    name: str
    Text for Label.
    """

    def __init__(self, name: str):
        super().__init__()
        self.label = QtWidgets.QLabel()
        self.label.setMaximumHeight(28)
        self.addWidget(self.label)
        self.name = name

    @property
    def name(self):
        """The name property returns the name of the node."""
        return self.label.text()

    @name.setter
    def name(self, string: str | None):
        """The name setter changes the text of the label."""
        if string:
            self.label.setText(string)


class EditField(BasicHLayout):
    """
    Settings EditField Layout with Label based on BasicLayout.

    Current content is stored in value for further use.

    Attributes:
    ----------
    string: str
        Default text for LineEdit
    value:
        Stores current value and is used for im and export.
    """

    def __init__(
        self,
        title: str | None = None,
        string: str | None = None,
        value_range: list | None = None,
    ):
        super().__init__(name=title)
        self.value_range = value_range
        # Add Spacer first
        spacer = QtWidgets.QSpacerItem(
            28,
            28,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.addSpacerItem(spacer)
        # Add Editfield
        self.editfield = QtWidgets.QLineEdit()
        self.editfield.setMaximumHeight(28)
        self.editfield.setMaximumWidth(48)
        self.editfield.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.addWidget(self.editfield)
        self.value = string

    @property
    def value(self):
        return self.editfield.text()

    @value.setter
    def value(self, string: str | None):
        if string:
            # The following needs to be moved to a method that handles ef changes
            if self.value_range:
                if string.isdigit():
                    value = int(string)
                    if value > self.value_range[1]:
                        value = self.value_range[1]
                    elif value < self.value_range[0]:
                        value = self.value_range[0]
                    self.editfield.setText(str(value))
            else:
                self.editfield.setText(string)


class ColorEditField(EditField):
    """
    Inheritance from normal Editfield with additional colorbox for display

    Methods:
    ----------
    color_changed(self):
        Sets color label background color for editfield signal
    """

    def __init__(self, title: str | None = None, string: str | None = None):
        super().__init__(title, string)
        self.colorbox = QtWidgets.QLabel()
        self.colorbox.setMaximumHeight(28)
        self.colorbox.setMaximumWidth(28)
        self.editfield.textChanged.connect(self.color_changed)
        self.editfield.setMaximumWidth(64)
        self.addWidget(self.colorbox)
        self.value = string

    @EditField.value.setter
    def value(self, string: str | None):
        if string and hasattr(self, "colorbox"):
            self.editfield.setText(string)
            self.colorbox.setStyleSheet(f"background-color: {string}")

    def color_changed(self):
        self.colorbox.setStyleSheet(f"background-color: {self.editfield.text()}")


class CheckBox(BasicHLayout):
    """Settings Checkbox Layout based on BasicLayout."""

    def __init__(self, title: str | None = None, state: bool | None = None):
        super().__init__(name=title)
        # Add Spacer first
        spacer = QtWidgets.QSpacerItem(
            28,
            28,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.addSpacerItem(spacer)
        # Add Checkbox
        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.setMaximumHeight(28)
        self.addWidget(self.checkbox)
        self.value = state

    @property  # Checkbox current state
    def value(self):
        return self.checkbox.isChecked()

    @value.setter
    def value(self, state: bool | None):
        self.checkbox.setChecked(state)


class ComboBox(BasicHLayout):
    """Settings Combobox Layout."""

    def __init__(self, name: str, default: str, items: list):
        super().__init__(name=name)
        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(items)
        self.addWidget(self.combobox)
        self.value = default
        self.items = items

    @property
    def value(self):
        return self.combobox.currentText()

    @value.setter
    def value(self, text: str):
        self.combobox.setCurrentText(text)


# noinspection PyUnresolvedReferences
class SettingsDlg(QtWidgets.QDialog):
    """
    Settings DLG window.

    Based on QDialog and a list of different sub layouts and Widgets for the different settings.

    Attributes:
    ----------
    settings_qt: QtCore.QSettings
        Dictionary from QSettings
    settings: dict
        Dictionary containing additional Settings beside the QSettings

    Methods:
    ----------
    get_settings_data(self, app_data):
        Return the settings back to main app
    """

    def __init__(
        self,
        # parent: QtWidgets.QMainWindow,
        settings_qt: QtCore.QSettings,
        settings: dict,
    ) -> None:
        super().__init__()

        # Basic Setup
        self.settings_dict = settings
        self.settings_qt = settings_qt
        self.setWindowTitle("Settings")
        img = Path(
            Path(__file__).parent.parent, "resources", "images", "settings.ico"
        ).__str__()  # HARDCODED
        self.setWindowIcon(QtGui.QIcon(img))
        self.setMinimumSize(192, 64)

        # Create main Layout
        self.main_layout = QtWidgets.QVBoxLayout()

        # General Settings
        self.main_layout.addLayout(TopicSeperator("general", "General:"))
        self.theme_combobox = ComboBox(
            "Theme:", settings_qt.value("theme", type=str), ["Dark", "Light"]
        )
        self.main_layout.addLayout(self.theme_combobox)

        self.multithreading_checkbox = CheckBox(
            "Multithreading:", self.settings_qt.value("multithreading", type=bool)
        )
        self.main_layout.addLayout(self.multithreading_checkbox)

        self.number_pools_ef = EditField(
            "Number of Pools",
            str(self.settings_qt.value("number_of_pools", type=int)),
            value_range=[0, cpu_count()],
        )
        self.main_layout.addLayout(self.number_pools_ef)

        # Segmentation Settings
        self.main_layout.addLayout(TopicSeperator("seg", "Segmentation Options:"))

        self.seg_color_efs = list()
        for idx, color in enumerate(self.settings_dict["seg_colors"]):
            color_ef = ColorEditField(f"#{idx}", color)
            self.seg_color_efs.append(color_ef)
            self.main_layout.addLayout(color_ef)

        self.seg_edge_alpha_ef = EditField(
            "Alpha:", str(self.settings_dict["seg_edge_alpha"])
        )
        self.main_layout.addLayout(self.seg_edge_alpha_ef)

        self.seg_line_width_ef = EditField(
            "Line Width:", str(self.settings_dict["seg_line_width"])
        )
        self.main_layout.addLayout(self.seg_line_width_ef)

        self.seg_face_alpha_ef = EditField(
            "Alpha:", str(self.settings_dict["seg_face_alpha"])
        )
        self.main_layout.addLayout(self.seg_face_alpha_ef)

        # CLOSE BUTTON
        button_layout = QtWidgets.QHBoxLayout()
        spacer = QtWidgets.QSpacerItem(
            28,
            28,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        button_layout.addSpacerItem(spacer)
        self.close_button = QtWidgets.QPushButton()
        self.close_button.setText("Close")
        self.close_button.setMaximumWidth(75)
        self.close_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )
        button_layout.addWidget(self.close_button)
        self.close_button.clicked.connect(self.accept)
        self.main_layout.addLayout(button_layout)

        self.setLayout(self.main_layout)

    def get_settings_data(self, app_data: AppData):
        """Get settings from SettingsDlg to main UI"""
        self.settings_qt.setValue("theme", self.theme_combobox.value)
        self.settings_qt.setValue("multithreading", self.multithreading_checkbox.value)
        self.settings_qt.setValue("number_of_pools", int(self.number_pools_ef.value))
        colors = list()
        for widget in self.seg_color_efs:
            colors.append(widget.value)
        app_data.plt["seg_colors"] = colors
        # Edge
        app_data.plt["seg_edge_alpha"] = float(
            self.seg_edge_alpha_ef.value.replace(",", ".")
        )
        self.settings_qt.setValue(
            "default_seg_edge_alpha",
            float(self.seg_edge_alpha_ef.value.replace(",", ".")),
        )
        app_data.plt["seg_line_width"] = float(self.seg_line_width_ef.value)
        self.settings_qt.setValue(
            "default_seg_line_width", float(self.seg_line_width_ef.value)
        )
        # Face
        app_data.plt["seg_face_alpha"] = float(
            self.seg_face_alpha_ef.value.replace(",", ".")
        )
        self.settings_qt.setValue(
            "default_seg_face_alpha",
            float(self.seg_face_alpha_ef.value.replace(",", ".")),
        )
        return self.settings_qt, app_data
