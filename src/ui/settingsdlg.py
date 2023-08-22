from PyQt6 import QtWidgets, QtCore, QtGui
from pathlib import Path


class TopicSeperator(QtWidgets.QVBoxLayout):
    def __init__(self, name: str, text: str):
        super().__init__()
        self.name = name
        general_label = QtWidgets.QLabel(text)
        self.addWidget(general_label)
        general_line = QtWidgets.QFrame()
        general_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        general_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.addWidget(general_line)


class EditField(QtWidgets.QHBoxLayout):
    def __init__(self, title: str | None = None, string: str | None = None):
        super().__init__()
        self.label = QtWidgets.QLabel()
        self.label.setMaximumHeight(28)
        self.editfield = QtWidgets.QLineEdit()
        self.editfield.setMaximumHeight(28)
        self.addWidget(self.label)
        self.addWidget(self.editfield)
        self.name = title
        self.value = string

    @property
    def name(self):
        return self.label.text()

    @name.setter
    def name(self, string: str | None):
        if string:
            self.label.setText(string)

    @property
    def value(self):
        return self.editfield.text()

    @value.setter
    def value(self, string: str | None):
        if string:
            self.editfield.setText(string)


class ColorEditField(EditField):
    def __init__(self, title: str | None = None, string: str | None = None):
        super().__init__(title, string)
        self.colorbox = QtWidgets.QLabel()
        self.colorbox.setMaximumHeight(28)
        self.colorbox.setMaximumWidth(28)
        self.editfield.textChanged.connect(self.color_changed)
        self.addWidget(self.colorbox)

        self.name = title
        self.value = string

    @EditField.value.setter
    def value(self, string: str | None):
        if string and hasattr(self, "colorbox"):
            self.editfield.setText(string)
            self.colorbox.setStyleSheet(f"background-color: {string}")

    def color_changed(self):
        self.colorbox.setStyleSheet(f"background-color: {self.editfield.text()}")


class SettingsDictionary(object):
    @staticmethod
    def get_settings_dict(fit_data):
        return {
            "plt.seg_colors": fit_data.plt.seg_colors,
            "plt.seg_alpha": fit_data.plt.seg_alpha,
        }


class SettingsWindow(QtWidgets.QDialog):
    def __init__(
        self,
        # parent: QtWidgets.QMainWindow,
        settings_qt: QtCore.QSettings,
        settings: dict,
    ) -> None:
        super().__init__()

        self.settings_dict = settings
        self.settings_qt = settings_qt
        self.setWindowTitle("Settings")
        img = Path(Path(__file__).parent, "resources", "Logo.png").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        # TODO Adjust Size automatically
        self.setMinimumSize(192, 64)

        self.main_layout = QtWidgets.QVBoxLayout()

        self.main_layout.addLayout(TopicSeperator("general", "General:"))

        self.theme_layout = QtWidgets.QHBoxLayout()
        theme_label = QtWidgets.QLabel("Theme:")
        self.theme_layout.addWidget(theme_label)
        self.theme_combobox = QtWidgets.QComboBox()
        self.theme_combobox.addItems(["Dark", "Light"])
        self.theme_combobox.setCurrentText(settings_qt.value("theme"))
        # self.theme_combobox.currentIndexChanged.connect(self._theme_changed)
        self.theme_layout.addWidget(self.theme_combobox)

        self.main_layout.addLayout(self.theme_layout)

        self.main_layout.addLayout(TopicSeperator("seg", "Segmentation Colors:"))

        self.seg_color_edit_fields = list()
        for idx, color in enumerate(self.settings_dict["plt.seg_colors"]):
            color_edit_field = ColorEditField(f"#{idx}", color)
            # color_edit_field.name =
            # color_edit_field.value = color
            self.seg_color_edit_fields.append(color_edit_field)
            self.main_layout.addLayout(color_edit_field)

        self.seg_alpha_edit_field = EditField(
            "Alpha", str(self.settings_dict["plt.seg_alpha"])
        )
        self.main_layout.addLayout(self.seg_alpha_edit_field)

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

    def get_settings_data(self, app_data):
        self.settings_qt.setValue("theme", self.theme_combobox.currentText())
        colors = list()
        for widget in self.seg_color_edit_fields:
            colors.append(widget.value)
        app_data.plt.seg_colors = colors
        app_data.plt.seg_alpha = float(self.seg_alpha_edit_field.value)
        return self.settings_qt, app_data  # , self.settings_dict

    # @settings_data.setter
    # def settings_data(self, settings_qt, settings_dict):
    #
