from __future__ import annotations
import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore
from typing import Callable

from PyQt6.QtWidgets import QGridLayout

from src.fit.parameters import Parameters, NNLSregParams, IVIMParams
from src.exceptions import ClassMismatch
from src.appdata import AppData

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.menubar import MenuBar


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
            FittingWidgets.WidgetData.__init__(
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
            FittingWidgets.WidgetData.__init__(
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
            FittingWidgets.WidgetData.__init__(
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
            FittingWidgets.WidgetData.__init__(
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


class BottomLayout(QtWidgets.QHBoxLayout):
    def __init__(self, parent: FittingDlg):
        super().__init__()
        self.parent = parent
        self.height = 28
        self.width = 28
        # Load Button
        self.load_button = QtWidgets.QPushButton()
        self.load_button.setIcon(
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
            )
        )
        self.load_button.setMinimumSize(self.width, self.height)
        self.load_button.setMaximumSize(self.width, self.height)
        self.load_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.load_button.clicked.connect(self.load_button_pushed)
        self.addWidget(self.load_button)
        # Save Button
        self.save_button = QtWidgets.QPushButton()
        self.save_button.setIcon(
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )
        self.save_button.setMinimumSize(self.width, self.height)
        self.save_button.setMaximumSize(self.width, self.height)
        self.save_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.save_button.clicked.connect(self.save_button_pushed)
        self.addWidget(self.save_button)
        # Spacer
        spacer = QtWidgets.QSpacerItem(
            self.height,
            self.height,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.addSpacerItem(spacer)
        # Accept Button
        self.accept_button = QtWidgets.QPushButton()
        self.accept_button.setText("Run Fitting")
        self.accept_button.setMaximumWidth(75)
        self.accept_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.accept_button.setMaximumHeight(self.height)
        self.addWidget(self.accept_button)
        self.accept_button.clicked.connect(self.accept_button_pushed)
        self.accept_button.setFocus()

    def accept_button_pushed(self):
        """Accept button callback"""
        # self.output_dict = dict()
        for key in self.parent.fit_dict:
            self.parent.fit_dict[key].current_value = self.parent.fit_dict[key].value
        self.parent.run = True
        self.parent.close()

    def load_button_pushed(self):
        """Load Json Button callback"""
        path = Path(
            QtWidgets.QFileDialog.getOpenFileName(
                caption="Open Parameter json File",
                directory=self.parent.app_data.last_dir.__str__(),
                filter=".json Files (*.json)",
            )[0]
        )
        if path.is_file():
            print(f"Loading parameters from {path}")
            try:
                self.parent.fit_params.load_from_json(path)

                if isinstance(self.parent.fit_params, NNLSregParams):
                    self.parent.fit_dict = FittingDictionaries.get_nnls_dict(
                        self.parent.fit_params
                    )
                elif isinstance(self.parent.fit_params, IVIMParams):
                    self.parent.fit_dict = FittingDictionaries.get_ivim_dict(
                        self.parent.fit_params
                    )
                # TODO: UI is not refreshing properly
                self.parent.setup_ui()
            except ClassMismatch:
                pass
            self.parent.app_data.last_dir = path.parent

    def save_button_pushed(self):
        path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                caption="Save Parameter json File",
                directory=self.parent.app_data.last_dir.__str__(),
                filter="All files (*.*);; JSON (*.json)",
                initialFilter="JSON (*.json)",
            )[0]
        )
        if not path.is_dir():
            self.parent.fit_params.save_to_json(path)
            self.parent.app_data.last_dir = path.parent


class FittingDlg(QtWidgets.QDialog):
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

    main_grid: QGridLayout
    bottom_layout: BottomLayout

    def __init__(
        self,
        name: str,
        fitting_dict: dict | None = None,
        fit_params: IVIMParams | NNLSregParams | None = None,
        app_data: AppData | None = None,
    ) -> None:
        """Main witting DLG window."""
        super().__init__()
        self.main_layout = None
        self.run = False
        self.name = name
        self.fit_dict = fitting_dict if not None else dict()
        self.fit_params = fit_params
        self.app_data = app_data
        self.setup_ui()

    def setup_ui(self):
        # Prepare Window
        self.setWindowTitle("Fitting " + self.name)
        self.setWindowIcon(
            QtGui.QIcon(
                Path(
                    Path(__file__).parent.parent.parent.parent,
                    "resources",
                    "images",
                    "PyneappleLogo.ico",
                ).__str__()
            )
        )
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
        self.main_layout.addLayout(self.main_grid)

        # Setup Parameter Fields for fitting
        self.load_widgets_from_dict()

        seperator_line = QtWidgets.QFrame()
        seperator_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        seperator_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.main_layout.addWidget(seperator_line)

        self.bottom_layout = BottomLayout(self)
        self.main_layout.addLayout(self.bottom_layout)
        self.bottom_layout.accept_button.setFocus()

    def refresh_ui_by_model_changed(self):
        # Get new model
        key = "n_components"
        widget = self.fit_dict[key]
        model = widget.currentText()
        # Unload Grid Layout
        self.remove_widgets(self.main_grid)
        # Recreate fit-dict
        self.fit_params.n_components = model
        self.fit_dict = FittingDictionaries.get_ivim_dict(self.fit_params)
        # Load Dict
        self.load_widgets_from_dict()

    @staticmethod
    def remove_widgets(layout: QtWidgets.QLayout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

    def load_widgets_from_dict(self):
        for idx, key in enumerate(self.fit_dict):
            label = QtWidgets.QLabel(self.fit_dict[key].name + ":")
            self.main_grid.addWidget(label, idx, 0)
            self.main_grid.addWidget(self.fit_dict[key], idx, 1)
            if key == "n_components":
                self.fit_dict[key].currentIndexChanged.connect(
                    self.refresh_ui_by_model_changed
                )

    def dict_to_attributes(self, fit_parameters: Parameters):
        # NOTE b_values and other special values have to be popped first
        for key, item in self.fit_dict.items():
            entries = key.split(".")
            if len(entries) == 2:
                # for parameter dicts
                c_dict = getattr(fit_parameters, entries[0], {})
                c_dict[entries[-1]] = item.value
                setattr(fit_parameters, entries[0], c_dict)
            else:
                setattr(fit_parameters, entries[-1], item.value)


class FittingDictionaries(object):
    """
    Collection of different basic fitting_dictionaries for the FittingDlg.

    Methods:
    ----------
    get_multi_exp_dict(fit_params: IVIMParams):
        Multi-exponential fitting parameters.
    get_nnls_dict(fit_params: NNLSregParams):
        NNLS fitting parameters.
    """

    @staticmethod
    def get_ivim_dict(fit_params: IVIMParams):
        models = ["MonoExp", "BiExp", "TriExp"]
        fit_dict = {
            "n_components": FittingWidgets.ComboBox(
                "Model",
                current_value=models[fit_params.n_components - 1],
                value_range=models,
                tooltip="Number of Components to fit",
            ),
            "fit_area": FittingWidgets.ComboBox(
                "Fitting Area", "Pixel", ["Pixel", "Segmentation"]
            ),
            "max_iter": FittingWidgets.EditField(
                "Maximum Iterations",
                fit_params.max_iter,
                [0, np.power(10, 6)],
                tooltip="Maximum number of iterations for the fitting algorithm",
            ),
            "boundaries.x0": FittingWidgets.EditField(
                "Start Values",
                fit_params.boundaries["x0"],
                None,
                tooltip="Start Values",
            ),
            "boundaries.lb": FittingWidgets.EditField(
                "Lower Boundaries",
                fit_params.boundaries["lb"],
                None,
                tooltip="Lower fitting Boundaries",
            ),
            "boundaries.ub": FittingWidgets.EditField(
                "Upper Boundaries",
                fit_params.boundaries["ub"],
                None,
                tooltip="Upper fitting Boundaries",
            ),
            "TM": FittingWidgets.EditField(
                "Mixing Time (TM)",
                current_value=fit_params.TM,
                value_range=[0, 10000],
                value_type=float,
                tooltip="Set Mixing Time if you want to perform advanced ADC fitting",
            ),
            "b_values": FittingWidgets.PushButton(
                name="Load B-Values",
                current_value=str(fit_params.b_values),
                button_function=FittingDictionaries._load_b_values,
                button_text="Open File",
            ),
        }
        return fit_dict

    @staticmethod
    def get_ideal_dict(fit_params: IVIMParams):
        models = ["MonoExp", "BiExp", "TriExp"]
        fit_dict = {
            "n_components": FittingWidgets.ComboBox(
                "Model",
                current_value=models[fit_params.n_components - 1],
                value_range=models,
                tooltip="Number of Components to fit",
            ),
            "max_iter": FittingWidgets.EditField(
                "Maximum Iterations",
                fit_params.max_iter,
                [0, np.power(10, 6)],
                tooltip="Maximum number of iterations for the fitting algorithm",
            ),
            "boundaries.x0": FittingWidgets.EditField(
                "Start Values",
                fit_params.boundaries["x0"],
                None,
                tooltip="Start Values",
            ),
            "boundaries.lb": FittingWidgets.EditField(
                "Lower Boundaries",
                fit_params.boundaries["lb"],
                None,
                tooltip="Lower fitting Boundaries",
            ),
            "boundaries.ub": FittingWidgets.EditField(
                "Upper Boundaries",
                fit_params.boundaries["ub"],
                None,
                tooltip="Upper fitting Boundaries",
            ),
            "b_values": FittingWidgets.PushButton(
                name="Load B-Values",
                current_value=str(fit_params.b_values),
                button_function=FittingDictionaries._load_b_values,
                button_text="Open File",
            ),
        }
        return fit_dict

    @staticmethod
    def get_nnls_dict(fit_params: NNLSregParams):
        return {
            "fit_area": FittingWidgets.ComboBox(
                "Fitting Area", "Pixel", ["Pixel", "Segmentation"]
            ),
            "max_iter": FittingWidgets.EditField(
                "Maximum Iterations",
                fit_params.max_iter,
                [0, np.power(10, 6)],
                tooltip="Maximum number of iterations for the fitting algorithm",
            ),
            "boundaries.n_bins": FittingWidgets.EditField(
                "Number of Bins",
                fit_params.boundaries["n_bins"],
                [0, np.power(10, 6)],
            ),
            "boundaries.d_range": FittingWidgets.EditField(
                "Diffusion Range",
                fit_params.boundaries["d_range"],
                [0, 1],
                tooltip="Number of exponential terms used for fitting",
            ),
            "reg_order": FittingWidgets.ComboBox(
                "Regularisation Order",
                str(fit_params.reg_order),
                ["0", "1", "2", "3", "CV"],
            ),
            "mu": FittingWidgets.EditField(
                "Regularisation Factor",
                fit_params.mu,
                [0.0, 1.0],
            ),
            "b_values": FittingWidgets.PushButton(
                name="Load B-Values",
                current_value=str(fit_params.b_values),
                button_function=FittingDictionaries._load_b_values,
                button_text="Open File",
            ),
        }

    @staticmethod
    def _load_b_values():
        path = QtWidgets.QFileDialog.getOpenFileName(
            caption="Open B-Value File",
            directory="",
        )[0]

        if path:
            file = Path(path)
            with open(file, "r") as f:
                # find away to decide which one is right
                # self.b_values = np.array([int(x) for x in f.read().split(" ")])
                b_values = [int(x) for x in f.read().split("\n")]
            return b_values
        else:
            return None
