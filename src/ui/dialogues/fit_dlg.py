from __future__ import annotations
import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore
from typing import Callable, TYPE_CHECKING

from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QMenuBar, QGridLayout

from src.appdata import AppData
from src.exceptions import ClassMismatch
import src.fit.parameters as params
from src.ui.dialogues.settings_dlg import TopicSeperator

if TYPE_CHECKING:
    from PyNeapple_UI import MainWindow


class SeperatorWidget(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


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
            current_value: int | float | np.ndarray | str = 1,
            value_range: list | None = None,
            value_type: type | None = None,
        ):
            self.current_value = current_value
            self.value_range = value_range if not None else list()
            # self.value_type = value_type if not None else type(current_value)
            if value_type is None:
                self.value_type = type(current_value)
            elif value_type is not None:
                self.value_type = value_type
            self.__value = current_value
            self.alignment_flag = None

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
            current_value: int | float | np.ndarray,
            value_range: list | None,
            value_type: type | None = None,
            tooltip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(
                self, current_value, value_range, value_type
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
            current_value: bool,
            value_range: list,
            value_type: type | None = None,
            tooltip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(
                self, current_value, value_range, value_type
            )
            QtWidgets.QCheckBox.__init__(self)
            # self.setText(str(current_value))
            self.stateChanged.connect(self._state_changed)
            if tooltip:
                self.setToolTip(tooltip)
            self.alignment_flag = QtCore.Qt.AlignmentFlag.AlignCenter
            self.setChecked(current_value)

        def _state_changed(self):
            self.value = self.isChecked()

    class ComboBox(WidgetData, QtWidgets.QComboBox):
        """QComboBox enhanced with WidgetData"""

        def __init__(
            self,
            current_value: str,
            value_range: list,
            value_type: type | None = None,
            tooltip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(
                self, current_value, value_range, value_type
            )
            QtWidgets.QComboBox.__init__(self)
            self.addItems(value_range)
            self.setCurrentText(current_value)
            self.currentIndexChanged.connect(self.__text_changed)
            if tooltip:
                self.setToolTip(tooltip)
            self.alignment_flag = None

        def __text_changed(self):
            self.value = self.currentText()

    class PushButton(WidgetData, QtWidgets.QPushButton):
        """
        QPushButton enhanced with WidgetData.

        Needs an additional callback function and button text.
        """

        def __init__(
            self,
            current_value: np.ndarray | str,
            value_type: type | None = None,
            button_function: Callable = None,
            button_text: str | None = None,
            tooltip: str | None = None,
        ):
            FittingWidgets.WidgetData.__init__(self, current_value, [], value_type)
            QtWidgets.QPushButton.__init__(self)
            self.value = current_value
            self.clicked.connect(lambda x: self.__button_clicked(button_function))
            if button_text:
                self.setText(button_text)
            if tooltip:
                self.setToolTip(tooltip)
            self.alignment_flag = QtCore.Qt.AlignmentFlag.AlignCenter

        def __button_clicked(self, button_function: Callable):
            self.value = button_function()


class FittingMenuBar(QtWidgets.QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.file_menu = QtWidgets.QMenu("&File", self)
        self.addMenu(self.file_menu)
        # Load Action
        self.load_action = QtGui.QAction()
        self.load_action.setText("Open...")
        self.load_action.setIcon(
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
            )
        )
        self.load_action.triggered.connect(self.load_action_pushed)
        self.file_menu.addAction(self.load_action)
        # Save Action
        self.save_action = QtGui.QAction()
        self.save_action.setText("Save...")
        self.save_action.setIcon(
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )
        self.save_action.triggered.connect(self.save_action_pushed)
        self.file_menu.addAction(self.save_action)

    def load_action_pushed(self):
        """Load Json Button callback"""
        path = Path(
            QtWidgets.QFileDialog.getOpenFileName(
                caption="Open Parameter json File",
                directory=self.parent.parent.data.last_dir.__str__(),
                filter=".json Files (*.json)",
            )[0]
        )
        if path.is_file():
            print(f"Loading parameters from {path}")
            # try:
            #     self.parent.fit_params.load_from_json(path)
            #
            #     if isinstance(self.parent.fit_params, NNLSregParams):
            #         self.parent.fit_dict = FittingDictionaries.get_nnls_dict(
            #             self.parent.fit_params
            #         )
            #     elif isinstance(self.parent.fit_params, IVIMParams):
            #         self.parent.fit_dict = FittingDictionaries.get_ivim_dict(
            #             self.parent.fit_params
            #         )
            #     # TODO: UI is not refreshing properly
            #     self.parent.setup_ui()
            # except ClassMismatch:
            #     pass
            # self.parent.app_data.last_dir = path.parent

    def save_action_pushed(self):
        path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                caption="Save Parameter json File",
                directory=self.parent.parent.data.last_dir.__str__(),
                filter="All files (*.*);; JSON (*.json)",
                initialFilter="JSON (*.json)",
            )[0]
        )
        if not path.is_dir():
            # self.parent.fit_params.save_to_json(path)
            self.parent.data.last_dir = path.parent


class ParameterLayout(QtWidgets.QGridLayout):
    def __init__(self, parent: BasicFittingDlg):
        super().__init__()
        self.parent = parent
        self.row_iterator = 0

        # FitArea
        self.fit_area = FittingWidgets.ComboBox("Pixel", ["Pixel", "Segmentation"])
        self.add_parameter("Fit Area:", self.fit_area)

        # Scale Image
        self.scale_image = FittingWidgets.CheckBox(
            current_value=True
            if self.parent.fit_params.scale_image == "S/S0"
            else False,
            value_range=[True, False],
            tooltip="Scale the image to first time point.",
        )
        self.add_parameter(
            "Scale Images:", self.scale_image, alignment=self.scale_image.alignment_flag
        )

        # B-Values
        self.b_values = FittingWidgets.PushButton(
            current_value=str(self.parent.fit_params.b_values),
            button_function=self._load_b_values,
            button_text=" Open File...",
            # tooltip= show bvalues while hovering
        )
        self.b_values.setIcon(
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
            )
        )
        self.add_parameter("Load B-Values:", self.b_values)

        # Max Iterations
        self.max_iterations = FittingWidgets.EditField(
            current_value=self.parent.fit_params.max_iter,
            value_range=[0, np.power(10, 6)],
            tooltip="Maximum number of iterations for the fitting algorithm",
        )
        self.add_parameter("Maximum Iterations:", self.max_iterations)

    def add_parameter(self, text: str, widget, **kwargs):
        self.addWidget(QtWidgets.QLabel(text), self.row_iterator, 0)
        if kwargs.get("alignment", None):
            self.addWidget(
                widget, self.row_iterator, 1, alignment=kwargs.get("alignment")
            )
        else:
            self.addWidget(widget, self.row_iterator, 1)
        self.row_iterator += 1

    def add_seperator(self):
        self.addWidget(SeperatorWidget(), self.row_iterator, 0, 1, 2)
        self.row_iterator += 1

    def _load_b_values(self):
        path = QtWidgets.QFileDialog.getOpenFileName(
            caption="Open B-Value File",
            directory=self.parent.parent.data.last_dir.__str__(),
        )[0]

        if path:
            file = Path(path)
            with open(file, "r") as f:
                # find away to decide which one is right
                # self.b_values = np.array([int(x) for x in f.read().split(" ")])
                b_values = [int(x) for x in f.read().split("\n")]
            self.values = b_values
        else:
            print("No B-Values loaded.")


class AcceptButtonLayout(QtWidgets.QHBoxLayout):
    def __init__(self, parent: BasicFittingDlg):
        super().__init__()
        self.parent = parent
        self.height = 28
        self.width = 28
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
        self.accept_button.setText("Run")
        self.accept_button.setMaximumWidth(75)
        self.accept_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.accept_button.setIcon(
            self.parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
            )
        )
        self.accept_button.setMaximumHeight(self.height)
        self.addWidget(self.accept_button)
        self.accept_button.clicked.connect(self.accept)
        self.accept_button.setFocus()

    def accept(self):
        """Accept button callback"""
        # self.output_dict = dict()
        # for key in self.parent.fit_dict:
        #     self.parent.fit_dict[key].current_value = self.parent.fit_dict[key].value
        self.parent.run = True
        self.parent.close()


class BasicFittingDlg(QtWidgets.QDialog):
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

    accept_button: AcceptButtonLayout
    main_layout: QVBoxLayout
    menu_bar: FittingMenuBar
    parameters: ParameterLayout

    def __init__(self, parent: MainWindow, fit_params: parameters.Parameters):
        """Main witting DLG window."""
        super().__init__()

        self.parent = parent
        self.main_window = parent
        self.data = parent.data  # needed?
        self.fit_params = fit_params
        self.setup_ui()

    @property
    def fit_params(self):
        return self._fit_params

    @fit_params.setter
    def fit_params(self, fit_params):
        self._fit_params = fit_params

    def setup_ui(self):
        self.setWindowTitle("Fitting")
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
        self.sizeHint()
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

        # Add MenuBar
        # self.menu_bar = FittingMenuBar(self)

        # Add MainLayout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        # self.main_layout.addSpacing(16)
        # general_line = QtWidgets.QFrame()
        # general_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        # general_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        # self.main_layout.addWidget(general_line)
        #
        self.parameters = ParameterLayout(self)
        self.main_layout.addLayout(self.parameters)
        self.main_layout.addWidget(SeperatorWidget())

        # self.advanced_parameters = QtWidgets.QGridLayout()
        # self.main_layout.addLayout(self.advanced_parameters)

        self.accept_button = AcceptButtonLayout(self)
        self.main_layout.addLayout(self.accept_button)


class IVIMFittingDlg(BasicFittingDlg):
    upper_boundaries: FittingWidgets.EditField
    lower_boundaries: FittingWidgets.EditField
    start_values: FittingWidgets.EditField
    fit_type: FittingWidgets.ComboBox

    def __init__(self, parent: MainWindow, fit_params: params.IVIMParams):
        super().__init__(parent, fit_params)
        self.models = ["MonoExp", "BiExp", "TriExp"]
        self._init_advanced_parameters()

    def _init_advanced_parameters(self):
        self.parameters.add_seperator()

        # Fitting Type // Number Components
        self.fit_type = FittingWidgets.ComboBox(
            current_value=self.models[self.fit_params.n_components - 1]
            if self.fit_params.n_components is not None
            else self.models[0],
            value_range=self.models,
            tooltip="Number of Components to fit",
        )
        self.fit_type.currentIndexChanged.connect(self._fit_type_changed)
        self.parameters.add_parameter("Fitting Type:", self.fit_type)

        # X0
        self.start_values = FittingWidgets.EditField(
            current_value=self.fit_params.boundaries["x0"],
            value_range=None,
            tooltip="Start Values",
        )
        self.parameters.add_parameter("Start Values:", self.start_values)

        # lb
        self.lower_boundaries = FittingWidgets.EditField(
            current_value=self.fit_params.boundaries["lb"],
            value_range=None,
            tooltip="Lower fitting Boundaries",
        )
        self.parameters.add_parameter("Lower Boundaries:", self.lower_boundaries)

        # ub
        self.upper_boundaries = FittingWidgets.EditField(
            current_value=self.fit_params.boundaries["ub"],
            value_range=None,
            tooltip="Upper fitting Boundaries",
        )
        self.parameters.add_parameter("Upper Boundaries:", self.upper_boundaries)

    def _fit_type_changed(self):
        print(self.fit_type.currentText())
        if self.fit_type.currentText() == self.models[0]:
            temp_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_IVIM_mono.json")
            )
            self.lower_boundaries.current_value = temp_params.boundaries["lb"]


class IDEALFittingDlg(IVIMFittingDlg):
    def __init__(self, parent: MainWindow, fit_params: params.IDEALParams):
        super().__init__(parent, fit_params)
        self._init_advanced_parameters()

    def _init_advanced_parameters(self):
        pass


class NNLSFittingDlg(BasicFittingDlg):
    def __init__(
        self,
        parent: MainWindow,
        fit_params: params.NNLSParams | params.NNLSregParams | params.NNLSregCVParams,
    ):
        super().__init__(parent, fit_params)
        self.reg_order_list = ["0", "1", "2", "3", "CV"]
        self._init_advanced_parameters()

    def _init_advanced_parameters(self):
        self.parameters.add_seperator()

        # Fitting Type // Regularisation Order
        self.reg_order = FittingWidgets.ComboBox(
            current_value=str(self.fit_params.reg_order),
            value_range=self.reg_order_list,
            tooltip="Regularisation Order or Cross Validation Approach",
        )
        self.parameters.add_parameter("Regularisation Order:", self.reg_order)

        # Number of Bins
        self.n_bins = FittingWidgets.EditField(
            current_value=self.fit_params.boundaries["n_bins"],
            value_range=[0, np.power(10, 6)],
            tooltip="Number of bins (Diffusion Components) to use for fitting",
        )
        self.parameters.add_parameter("Number of Bins:", self.n_bins)

        # Diffusion Range
        self.d_range = FittingWidgets.EditField(
            current_value=self.fit_params.boundaries["d_range"],
            value_range=[0, 1],
            tooltip="Diffusion Range to place bins in",
        )
        self.parameters.add_parameter("Diffusion Range:", self.d_range)

        # Regularisation Factor mu
        self.reg_factor = FittingWidgets.EditField(
            current_value=self.fit_params.mu,
            value_range=[0.0, 1.0],
            tooltip="Regularisation factor mu for different Regularisation Orders. \nNot for Cross Validation Approach.",
        )
        self.parameters.add_parameter("Regularisation Factor:", self.reg_factor)

        # Cross Validation Tolerance
        self.reg_cv_tol = FittingWidgets.EditField(
            current_value=getattr(self.fit_params, "tol")
            if hasattr(self.fit_params, "tol")
            else 0.0001,
            value_range=[0.0, 1.0],
            tooltip="Tolerance for Cross Validation Regularisation",
        )
        self.parameters.add_parameter("CV Tolerance:", self.reg_cv_tol)
