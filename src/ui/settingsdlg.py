from PyQt6 import QtWidgets, QtCore, QtGui
from pathlib import Path


class EditField(QtWidgets.QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.label = QtWidgets.QLabel()
        self.label.setMaximumHeight(28)
        self.editfield = QtWidgets.QLineEdit()
        self.editfield.setMaximumHeight(28)
        self.addWidget(self.label)
        self.addWidget(self.editfield)

    @property
    def name(self):
        return self.label.text()

    @name.setter
    def name(self, string: str):
        self.label.setText(string)

    @property
    def value(self):
        return self.editfield.text()

    @value.setter
    def value(self, string: str):
        self.editfield.setText(string)


class ColorEditField(EditField):
    def __init__(self):
        super().__init__()
        self.colorbox = QtWidgets.QLabel()
        self.colorbox.setMaximumHeight(28)
        self.colorbox.setMaximumWidth(28)
        self.editfield.textChanged.connect(self.color_changed)
        self.addWidget(self.colorbox)

    @EditField.value.setter
    def value(self, string: str):
        self.editfield.setText(string)
        self.colorbox.setStyleSheet(f"background-color: {string}")

    def color_changed(self):
        self.colorbox.setStyleSheet(f"background-color: {self.editfield.text()}")


class SettingsDictionary(object):
    @staticmethod
    def get_settings_dict(fit_data):
        return {"plt.seg_colors": fit_data.plt.seg_colors}


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
        self.theme_combobox.setCurrentText(settings_qt.value("theme"))
        # self.theme_combobox.currentIndexChanged.connect(self._theme_changed)
        self.theme_layout.addWidget(self.theme_combobox)

        self.main_layout.addLayout(self.theme_layout)

        segmentation_label = QtWidgets.QLabel("Segmentation Colors:")
        self.main_layout.addWidget(segmentation_label)
        segmentation_line = QtWidgets.QFrame()
        segmentation_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        segmentation_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.main_layout.addWidget(segmentation_line)
        self.seg_color_edit_fields = list()
        # default_colors = qt_settings.value("default_seg_colors", type=list)
        # default_colors = ["#ff0000", "#0000ff", "#00ff00", "#ffff00"]
        for idx, color in enumerate(self.settings_dict["plt.seg_colors"]):
            color_edit_field = ColorEditField()
            color_edit_field.name = f"#{idx}"
            color_edit_field.value = color
            self.seg_color_edit_fields.append(color_edit_field)
            self.main_layout.addLayout(color_edit_field)

        self.setLayout(self.main_layout)

    # def _theme_changed(self):
    #     current_style = QtWidgets.QApplication.style().objectName()
    #     new_style = "Fusion" if current_style != "Fusion" else "Windows"
    #     QtWidgets.QApplication.setStyle(new_style)

    def get_settings_data(self, app_data):
        self.settings_qt.setValue("theme", self.theme_combobox.currentText())
        app_data.plt.seg_colors = self.settings_dict["plt.seg_colors"]
        return self.settings_qt, app_data  # , self.settings_dict

    # @settings_data.setter
    # def settings_data(self, settings_qt, settings_dict):
    #
