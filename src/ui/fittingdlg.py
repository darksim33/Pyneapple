import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore
from typing import Callable

from src.fit.parameters import Parameters, NNLSregParams, MultiExpParams


class FittingDlg:
    class FittingWidgets(object):
        class WidgetData:
            """
            Basic widget enhancement class. To set up different dlg Widgets in the same way.

            Attributes:
            ----------
            name: str
                Name of the Widget and text that is displayed on the dlg ':' is  added separately
            current_value: int | float | np.ndarray | str
                The value the widget currently hold
            value_range: list
                Range of allowed Values
            value_type: Class | None = None
                Defines the value type of the handled variable.
                If the type is not defined here the input type of current_value will be used.
            tooltip: str | None = None
                Widget tooltip text
            value: @Property
                Can hold different types of classes to report back to main UI

            """

            def __init__(
                    self,
                    name: str = "",
                    current_value: int | float | np.ndarray | str = 1,
                    value_range: list | None = None,
                    value_type: type | None = None,
            ):
                self.name = name
                self.current_value = current_value
                self.value_range = value_range if not None else list()
                # self.value_type = value_type if not None else type(current_value)
                if value_type is None:
                    self.value_type = type(current_value)
                elif value_type is not None:
                    self.value_type = value_type
                self.__value = current_value

            @property
            def value(self):
                return self.__value

            @value.setter
            def value(self, arg):
                if type(arg) == str:
                    if self.value_type in [int, float]:
                        if arg.isdigit():
                            arg = self.value_type(arg)
                        else:
                            arg = None
                    elif self.value == np.ndarray:
                        arg = np.frombuffer(arg)
                    elif (not arg or arg == "None") and self.value_type is None:
                        arg = None

                # TODO: range implementation
                # if value < self.value_range[0] or value > self.value_range[1]:
                #     self.__value = self.default
                #     print("Value exceeded value range.")
                # else:
                #     self.__value = value
                self.__value = arg

        class EditField(WidgetData, QtWidgets.QLineEdit):
            """QLineEdit enhanced with WidgetData"""

            def __init__(
                    self,
                    name: str,
                    current_value: int | float | np.ndarray,
                    value_range: list | None,
                    value_type: type | None = None,
                    tooltip: str | None = None,
            ):
                FittingDlg.FittingWidgets.WidgetData.__init__(
                    self, name, current_value, value_range, value_type
                )
                QtWidgets.QLineEdit.__init__(self)
                self.setText(str(current_value))
                self.textChanged.connect(self._text_changed)
                self.setMaximumHeight(28)
                self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
                if tooltip:
                    self.setToolTip(tooltip)

            def _text_changed(self):
                self.value = self.text()

        class CheckBox(WidgetData, QtWidgets.QCheckBox):
            """QCheckbox enhanced with WidgetData"""

            def __init__(
                    self,
                    name: str,
                    current_value: int | float | np.ndarray,
                    value_range: list,
                    value_type: type | None = None,
                    tooltip: str | None = None,
            ):
                FittingDlg.FittingWidgets.WidgetData.__init__(
                    self, name, current_value, value_range, value_type
                )
                QtWidgets.QCheckBox.__init__(self)
                self.setText(str(current_value))
                self.stateChanged.connect(self._state_changed)
                if tooltip:
                    self.setToolTip(tooltip)

            def _state_changed(self):
                self.value = self.isChecked()

        class ComboBox(WidgetData, QtWidgets.QComboBox):
            """QComboBox enhanced with WidgetData"""

            def __init__(
                    self,
                    name: str,
                    current_value: str,
                    value_range: list,
                    value_type: type | None = None,
                    tooltip: str | None = None,
            ):
                FittingDlg.FittingWidgets.WidgetData.__init__(
                    self, name, current_value, value_range, value_type
                )
                QtWidgets.QComboBox.__init__(self)
                self.addItems(value_range)
                self.setCurrentText(current_value)
                self.currentIndexChanged.connect(self.__text_changed)
                if tooltip:
                    self.setToolTip(tooltip)

            def __text_changed(self):
                self.value = self.currentText()

        class ModelComboBox(ComboBox):
            def __init__(
                    self,
                    parent,
                    name: str,
                    current_value: str,
                    value_range: list,
                    value_type: type | None = None,
                    tooltip: str | None = None,
            ):
                super().__init__(
                    name,
                    current_value,
                    value_range,
                    value_type,
                    tooltip,
                )

        class PushButton(WidgetData, QtWidgets.QPushButton):
            """
            QPushButton enhanced with WidgetData.
            Needs an additional callback function and button text.
            """

            def __init__(
                    self,
                    name: str,
                    current_value: np.ndarray | str,
                    value_type: type | None = None,
                    button_function: Callable = None,
                    button_text: str | None = None,
                    tooltip: str | None = None,
            ):
                FittingDlg.FittingWidgets.WidgetData.__init__(
                    self, name, current_value, [], value_type
                )
                QtWidgets.QPushButton.__init__(self)
                self.value = current_value
                self.clicked.connect(lambda x: self.__button_clicked(button_function))
                if button_text:
                    self.setText(button_text)
                if tooltip:
                    self.setToolTip(tooltip)

            def __button_clicked(self, button_function: Callable):
                self.value = button_function()

    class Dialog(QtWidgets.QDialog):
        """
        Main witting DLG window.
        QDialog with some basic actions which are similar to all fitting methods and a dictionary containing identifiers and
        QWidget based FittingWidgets

        Attributes:
        ----------
        name: str
            Dlg name
        fitting_dict: dict
            Dictionary containing name keys and FittingWidgets.
            The name keys should align to Positions in fit_data.
            Currently only Widgets are allowed and Layouts are unsupported.

        Methods:
        ----------
        accept_button_pushed(self):
            Accept Button Method. Refreshing the fitting_dict with the newly assigned values.
        dict_to_attributes(self, fit_data: Parameters):
            Transforms dictionary entries to fit.Parameters Attributes.
            Dot indexing will be taken into account.
        """
        def __init__(self, name: str, fitting_dict: dict | None = None) -> None:
            super().__init__()
            self.run = False
            self.name = name
            self.fitting_dict = fitting_dict if not None else dict()
            self._setup_ui()

        def _setup_ui(self):
            # Prepare Window
            self.setWindowTitle("Fitting " + self.name)
            img = Path(Path(__file__).parent, "resources", "Logo.png").__str__()
            self.setWindowIcon(QtGui.QIcon(img))
            self.setMinimumSize(192, 64)
            self.setSizePolicy(
                QtWidgets.QSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Fixed,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
            )

            # Load main Parameter Widgets
            self.main_layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.main_layout)
            self.main_grid = QtWidgets.QGridLayout()

            if self.name == ("multiExp" or "IVIM"):
                self.main_grid.addWidget(QtWidgets.QLabel("Model:"))

            for idx, key in enumerate(self.fitting_dict):
                label = QtWidgets.QLabel(self.fitting_dict[key].name + ":")
                self.main_grid.addWidget(label, idx, 0)
                self.main_grid.addWidget(self.fitting_dict[key], idx, 1)
            self.main_layout.addLayout(self.main_grid)

            # Add accept Button
            button_layout = QtWidgets.QHBoxLayout()
            spacer = QtWidgets.QSpacerItem(
                28,
                28,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            button_layout.addSpacerItem(spacer)
            self.accept_button = QtWidgets.QPushButton()
            self.accept_button.setText("Run Fitting")
            self.accept_button.setMaximumWidth(75)
            self.accept_button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
            )
            self.accept_button.setMaximumHeight(28)
            button_layout.addWidget(self.accept_button)
            self.accept_button.clicked.connect(self.accept_button_pushed)
            self.accept_button.setFocus()
            self.main_layout.addLayout(button_layout)

        def accept_button_pushed(self):
            # self.output_dict = dict()
            for key in self.fitting_dict:
                self.fitting_dict[key].current_value = self.fitting_dict[key].value
            self.run = True
            self.close()

        def dict_to_attributes(self, fit_parameters: Parameters):
            # NOTE b_values and other special values have to be popped first
            self.fitting_dict["dict_type"].pop()
            for key, item in self.fitting_dict.items():
                entries = key.split(".")
                current_obj = fit_parameters
                if len(entries) > 1:
                    for entry in entries[:-1]:
                        current_obj = getattr(current_obj, entry)
                setattr(current_obj, entries[-1], item.value)

    class FittingDictionaries(object):
        """
        Collection of different basic fitting_dictionaries for the FittingDlg.

        Methods:
        ----------
        get_multi_exp_dict(fit_params: MultiExpParams):
            Multi-exponential fitting parameters.
        get_nnls_dict(fit_params: NNLSregParams):
            NNLS fitting parameters.
        """

        @staticmethod
        def get_multi_exp_dict(fit_params: MultiExpParams):
            models = ["MonoExp", "BiExp", "TriExp"]
            return {
                "n_components": FittingDlg.FittingWidgets.ComboBox(
                    "Model",
                    current_value=models[fit_params.n_components - 1],
                    value_range=models,
                    tooltip="Number of Components to fit",
                ),
                "fit_area": FittingDlg.FittingWidgets.ComboBox(
                    "Fitting Area", "Pixel", ["Pixel", "Segmentation"]
                ),
                "max_iter": FittingDlg.FittingWidgets.EditField(
                    "Maximum Iterations",
                    fit_params.max_iter,
                    [0, np.power(10, 6)],
                    tooltip="Maximum number of iterations for the fitting algorithm",
                ),
                "boundaries.x0": FittingDlg.FittingWidgets.EditField(
                    "Start Values", fit_params.boundaries.x0, None, tooltip="Start Values"
                ),
                "boundaries.lb": FittingDlg.FittingWidgets.EditField(
                    "Lower Boundaries",
                    fit_params.boundaries.lb,
                    None,
                    tooltip="Lower fitting Boundaries",
                ),
                "boundaries.ub": FittingDlg.FittingWidgets.EditField(
                    "Upper Boundaries",
                    fit_params.boundaries.ub,
                    None,
                    tooltip="Upper fitting Boundaries",
                ),
                "TM": FittingDlg.FittingWidgets.EditField(
                    "Mixing Time (TM)",
                    current_value=fit_params.TM,
                    value_range=[0, 10000],
                    value_type=float,
                    tooltip="Set Mixing Time if you want to perform advanced ADC fitting",
                ),
            }

        @staticmethod
        def get_nnls_dict(fit_params: NNLSregParams):
            return {
                "fit_area": FittingDlg.FittingWidgets.ComboBox(
                    "Fitting Area", "Pixel", ["Pixel", "Segmentation"]
                ),
                "max_iter": FittingDlg.FittingWidgets.EditField(
                    "Maximum Iterations",
                    fit_params.max_iter,
                    [0, np.power(10, 6)],
                    tooltip="Maximum number of iterations for the fitting algorithm",
                ),
                "boundaries.n_bins": FittingDlg.FittingWidgets.EditField(
                    "Number of Bins",
                    fit_params.boundaries.n_bins,
                    [0, np.power(10, 6)],
                ),
                "boundaries.d_range": FittingDlg.FittingWidgets.EditField(
                    "Diffusion Range",
                    fit_params.boundaries.d_range,
                    [0, 1],
                    tooltip="Number of exponential terms used for fitting",
                ),
                "reg_order": FittingDlg.FittingWidgets.ComboBox(
                    "Regularisation Order", "0", ["0", "1", "2", "3", "CV"]
                ),
                "mu": FittingDlg.FittingWidgets.EditField(
                    "Regularisation Factor",
                    fit_params.mu,
                    [0.0, 1.0],
                ),
            }
