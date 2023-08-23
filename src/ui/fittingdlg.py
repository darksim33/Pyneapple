import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtGui
from typing import Callable

from src.fit.parameters import Parameters


class FittingWidgets(object):
    class WidgetData:
        def __init__(
            self,
            name: str = "",
            current_value: int | float | np.ndarray = 1,
            value_range: list | None = None,
        ):
            self.name = name
            self.current_value = current_value
            self.value_range = value_range if not None else list()
            self.__value = current_value

        @property
        def value(self):
            return self.__value

        @value.setter
        def value(self, arg):
            if type(self.current_value) == (int or float):
                arg = type(self.current_value)(arg)
            elif type(arg) == np.ndarray:
                arg = np.frombuffer(arg)
            # TODO: range implementation
            # if value < self.value_range[0] or value > self.value_range[1]:
            #     self.__value = self.default
            #     print("Value exceded value range.")
            # else:
            #     self.__value = value
            self.__value = arg

    class EditField(WidgetData, QtWidgets.QLineEdit):
        def __init__(
            self,
            name: str,
            current_value: int | float | np.ndarray,
            value_range: list | None,
            tooltip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(self, name, current_value, value_range)
            QtWidgets.QLineEdit.__init__(self)
            self.setText(str(current_value))
            self.textChanged.connect(self._text_changed)
            self.setMaximumHeight(28)

        def _text_changed(self):
            self.value = self.text()

    class CheckBox(WidgetData, QtWidgets.QCheckBox):
        def __init__(
            self,
            name: str,
            current_value: int | float | np.ndarray,
            value_range: list,
            tooltip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(self, name, current_value, value_range)
            QtWidgets.QTextEdit.__init__(self)
            self.setText(str(current_value))
            self.stateChanged.connect(self._state_changed)

        def _state_changed(self):
            self.data.value = self.isChecked

    class ComboBox(WidgetData, QtWidgets.QComboBox):
        def __init__(
            self,
            name: str,
            current_value: str,
            value_range: list,
            tooltip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(self, name, current_value, value_range)
            QtWidgets.QTextEdit.__init__(self)
            self.addItems(value_range)
            self.setCurrentText(current_value)
            self.currentIndexChanged.connect(self.__text_changed)

        def __text_changed(self):
            self.value = self.currentText()

    class PushButton(WidgetData, QtWidgets.QPushButton):
        def __init__(
            self,
            name: str,
            current_value: np.ndarray | str,
            bttn_function: Callable = None,
            bttn_text: str | None = None,
            tootip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(self, name, current_value, [])
            QtWidgets.QPushButton.__init__(self)
            self.value = current_value
            self.clicked.connect(lambda x: self.__button_clicked(bttn_function))
            if bttn_text:
                self.setText(bttn_text)

        def __button_clicked(self, bttn_function: Callable):
            self.value = bttn_function()


class FittingWindow(QtWidgets.QDialog):
    def __init__(self, name: str, fitting_dict: dict) -> None:
        super().__init__()
        self.run = False
        self.fitting_dict = dict()
        self.fitting_dict = fitting_dict
        self.setWindowTitle("Fitting " + name)
        img = Path(Path(__file__).parent, "resources", "Logo.png").__str__()
        self.setWindowIcon(QtGui.QIcon(img))
        self.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
        )
        # self.setWindowIcon(QtGui.QIcon(img))
        self.setMinimumSize(192, 64)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_grid = QtWidgets.QGridLayout()
        for idx, key in enumerate(fitting_dict):
            # self.main_layout.addLayout(fitting_dict[key])
            label = QtWidgets.QLabel(self.fitting_dict[key].name + ":")
            self.main_grid.addWidget(label, idx, 0)
            self.main_grid.addWidget(self.fitting_dict[key], idx, 1)
        self.main_layout.addLayout(self.main_grid)

        button_layout = QtWidgets.QHBoxLayout()
        spacer = QtWidgets.QSpacerItem(
            28,
            28,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        button_layout.addSpacerItem(spacer)
        self.run_button = QtWidgets.QPushButton()
        self.run_button.setText("Run Fitting")
        self.run_button.setMaximumWidth(75)
        self.run_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )
        button_layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.run_button_pushed)
        self.main_layout.addLayout(button_layout)
        self.setLayout(self.main_layout)

    def run_button_pushed(self):
        # self.output_dict = dict()
        for key in self.fitting_dict:
            self.fitting_dict[key].current_value = self.fitting_dict[key].value
        self.run = True
        self.close()

    # NOTE: Still necessary?
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        return super().closeEvent(event)

    def dict_to_attributes(self, fit_data: Parameters):
        # NOTE b_values and other special values have to be poped first

        for key, item in self.fitting_dict.items():
            entries = key.split(".")
            current_obj = fit_data
            if len(entries) > 1:
                for entry in entries[:-2]:
                    current_obj = getattr(current_obj, entry)
            setattr(current_obj, entries[-1], item.value)


class FittingDictionaries(object):
    @staticmethod
    def get_mono_dict(fit_data):
        return {
            "fit_area": FittingWidgets.ComboBox(
                "Fitting Area", "Pixel", ["Pixel", "Segmentation"]
            ),
            "max_iter": FittingWidgets.EditField(
                "Maximum Iterations",
                fit_data.fit_params.max_iter,
                [0, np.power(10, 6)],
            ),
            "boundaries.x0": FittingWidgets.EditField(
                "Start Values",
                fit_data.fit_params.boundaries.x0,
                None,
            ),
            "boundaries.lb": FittingWidgets.EditField(
                "Lower Boundaries",
                fit_data.fit_params.boundaries.lb,
                None,
            ),
            "boundaries.ub": FittingWidgets.EditField(
                "Upper Boundaries",
                fit_data.fit_params.boundaries.ub,
                None,
            ),
        }

    @staticmethod
    def get_multiExp_dict(fit_data):
        return {
            "fit_area": FittingWidgets.ComboBox(
                "Fitting Area", "Pixel", ["Pixel", "Segmentation"]
            ),
            "max_iter": FittingWidgets.EditField(
                "Maximum Iterations",
                fit_data.fit_params.max_iter,
                [0, np.power(10, 6)],
            ),
            "boundaries.x0": FittingWidgets.EditField(
                "Start Values",
                fit_data.fit_params.boundaries.x0,
                None,
            ),
            "boundaries.lb": FittingWidgets.EditField(
                "Lower Boundaries",
                fit_data.fit_params.boundaries.lb,
                None,
            ),
            "boundaries.ub": FittingWidgets.EditField(
                "Upper Boundaries",
                fit_data.fit_params.boundaries.ub,
                None,
            ),
            "n_components": FittingWidgets.EditField(
                "Number of components",
                fit_data.fit_params.n_components,
                [0, 10],
            ),
        }

    @staticmethod
    def get_nnls_dict(fit_data):
        return {
            "fit_area": FittingWidgets.ComboBox(
                "Fitting Area", "Pixel", ["Pixel", "Segmentation"]
            ),
            "max_iter": FittingWidgets.EditField(
                "Maximum Iterations",
                fit_data.fit_params.max_iter,
                [0, np.power(10, 6)],
            ),
            "boundaries.n_bins": FittingWidgets.EditField(
                "Number of Bins",
                fit_data.fit_params.boundaries.n_bins,
                [0, np.power(10, 6)],
            ),
            "boundaries.d_range": FittingWidgets.EditField(
                "Diffusion Range",
                fit_data.fit_params.boundaries.d_range,
                [0, 1],
            ),
            "reg_order": FittingWidgets.ComboBox(
                "Regularisation Order", "0", ["0", "1", "2", "3", "CV"]
            ),
            "mu": FittingWidgets.EditField(
                "Regularisation Factor",
                fit_data.fit_params.mu,
                [0.0, 1.0],
            ),
        }
