from __future__ import annotations
import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore
from typing import Callable, TYPE_CHECKING
from abc import abstractmethod

from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QMenuBar, QGridLayout
from src.ui.dialogues import fitting_widgets

from src.appdata import AppData
from src.exceptions import ClassMismatch
import src.fit.parameters as params

if TYPE_CHECKING:
    from PyNeapple_UI import MainWindow


class SeperatorWidget(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


class FittingMenuBar(QtWidgets.QVBoxLayout):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.file_button = QtWidgets.QPushButton()
        self.file_button.setText("&File...")
        self.file_button.setShortcut("Ctrl+F")
        self.file_button.height = 18
        self.file_button.setMaximumWidth(50)
        # self.file_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.file_button.setStyleSheet("QPushButton{border: None}")
        # Should work but currently doesn't in combination with the border
        # NOTE: needs further refinement
        # self.file_button.setStyleSheet("QPushButton::menu-indicator {image: None;}")
        # self.file_button.setStyleSheet(
        #     "QPushButton{border: None}" "QPushButton::menu-indicator {image: None;}"
        # )
        self.addWidget(self.file_button)

        # Add Menu to button
        self.file_menu = QtWidgets.QMenu("&File")
        self.file_button.setMenu(self.file_menu)
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
            try:
                self.parent.fit_params.load_from_json(path)
                # TODO: Need new json_loader to switch dialog options
                # Add check and prompt for changed parameter set
                # Add refresh for layout
                if isinstance(self.parent.fit_params, params.IVIMParams):
                    self.parent.parameters = IVIMParameterLayout(self.parent)
                elif isinstance(self.parent.fit_params,
                                (params.NNLSParams, params.NNLSregParams, params.NNLSregCVParams)):
                    self.parent.parameters = NNLSParameterLayout(self.parent)
                elif isinstance(self.parent.fit_params, params.IVIMParams):
                    self.parent.parameters = IDEALParameterLayout(self.parent)
                else:
                    self.parent.parameters = ParameterLayout(self.parent)
            except ClassMismatch:
                print("Warning: No supported Parameter Class detected!")
            self.parent.parent.data.last_dir = path.parent

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
    def __init__(self, parent: FittingDlg):
        super().__init__()
        self.parent = parent
        self.row_iterator = 0

        self.add_seperator()

        # FitArea
        self.fit_area = fitting_widgets.ComboBox(
            value="Pixel", range_=["Pixel", "Segmentation"], dtype=str
        )
        self.add_parameter("Fit Area:", self.fit_area)

        # Scale Image
        self.scale_image = fitting_widgets.CheckBox(
            value=True if self.parent.fit_params.scale_image == "S/S0" else False,
            range_=[True, False],
            dtype=bool,
            tooltip="Scale the image to first time point.",
        )
        self.add_parameter(
            "Scale Images:", self.scale_image, alignment=self.scale_image.alignment_flag
        )

        # B-Values
        self.b_values = fitting_widgets.PushButton(
            current_value=str(self.parent.fit_params.b_values),
            button_function=self._load_b_values,
            button_text=" Open File...",
            dtype=np.ndarray,
            # tooltip= show b-values while hovering
        )
        self.b_values.setIcon(
            parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
            )
        )
        self.b_values.setToolTip("B-Values: " + self.b_values.value.__str__())
        self.add_parameter("Load B-Values:", self.b_values)

        # Max Iterations
        self.max_iterations = fitting_widgets.EditField(
            value=self.parent.fit_params.max_iter,
            range_=[0, np.power(10, 6)],
            dtype=int,
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
            self.value = b_values

            # set tooltip
            self.b_values.setToolTip("B-Values: " + b_values.__str__())
        else:
            print("No B-Values loaded.")

    def get_parameters(self) -> params.Parameters:
        self.parent.fit_params.fit_area = self.fit_area.value
        self.parent.fit_params.scale_image = self.scale_image.value
        self.parent.fit_params.b_values = self.b_values.value
        self.parent.fit_params.max_iter = self.max_iterations.value
        return self.parent.fit_params


class IVIMParameterLayout(ParameterLayout):
    def __init__(self, parent: FittingDlg):
        super().__init__(parent)
        self.parent.setWindowTitle("Fitting: IVIM")
        self.models = ["MonoExp", "BiExp", "TriExp"]
        self._init_advanced_parameters()

    def _init_advanced_parameters(self):
        self.add_seperator()

        # Fitting Type // Number
        # TODO: Number of components needs further refinement
        self.fit_type = fitting_widgets.ComboBox(
            value=(
                self.models[self.parent.fit_params.n_components - 1]
                if self.parent.fit_params.n_components is not None
                else self.models[0]
            ),
            range_=self.models,
            dtype=int,
            tooltip="Number of Components to fit",
        )
        self.fit_type.currentIndexChanged.connect(self._fit_type_changed)
        self.add_parameter("Fitting Type:", self.fit_type)

        self._init_boundaries()

    def _init_boundaries(self):
        # X0
        self.start_values = fitting_widgets.EditField(
            value=self.parent.fit_params.boundaries["x0"],
            range_=None,
            dtype=np.ndarray,
            tooltip="Start Values",
        )
        self.add_parameter("Start Values:", self.start_values)

        # lb
        self.lower_boundaries = fitting_widgets.EditField(
            value=self.parent.fit_params.boundaries["lb"],
            range_=None,
            dtype=np.ndarray,
            tooltip="Lower fitting Boundaries",
        )
        self.add_parameter("Lower Boundaries:", self.lower_boundaries)

        # ub
        self.upper_boundaries = fitting_widgets.EditField(
            value=self.parent.fit_params.boundaries["ub"],
            range_=None,
            dtype=np.ndarray,
            tooltip="Upper fitting Boundaries",
        )
        self.add_parameter("Upper Boundaries:", self.upper_boundaries)

    def get_parameters(self) -> params.IVIMParams:
        super().get_parameters()
        self.parent.fit_params.boundaries["x0"] = self.start_values.value
        self.parent.fit_params.boundaries["lb"] = self.lower_boundaries.value
        self.parent.fit_params.boundaries["ub"] = self.upper_boundaries.value
        return self.parent.fit_params

    def _fit_type_changed(self):
        if self.fit_type.currentText() == self.models[0]:
            self.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_IVIM_mono.json")
            )
        elif self.fit_type.currentText() == self.models[1]:
            self.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_IVIM_bi.json")
            )
        elif self.fit_type.currentText() == self.models[2]:
            self.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_IVIM_tri.json")
            )
        else:
            print("Selected model didn't fit to any listed Models.")
            return

        self.start_values.value = self.fit_params.boundaries["x0"]
        self.lower_boundaries.value = self.fit_params.boundaries["lb"]
        self.upper_boundaries.value = self.fit_params.boundaries["ub"]


class IDEALParameterLayout(IVIMParameterLayout):
    def __init__(self, parent: FittingDlg):
        super().__init__(parent)
        self.parent.setWindowTitle("Fitting: IDEAL")
        self.models = ["BiExp", "TriExp"]
        self._init_advanced_parameters()

    def _fit_type_changed(self):
        if self.fit_type.currentText() == self.models[0]:
            self.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_ideal_bi.json")
            )
        elif self.fit_type.currentText() == self.models[1]:
            self.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_ideal_tri.json")
            )
        else:
            print("Selected model didn't fit to any listed Models.")
            return

        self.start_values.value = self.fit_params.boundaries["x0"]
        self.lower_boundaries.value = self.fit_params.boundaries["lb"]
        self.upper_boundaries.value = self.fit_params.boundaries["ub"]

        # self.refresh_ui()

    def _init_advanced_parameters(self):
        self.add_seperator()

        # Fitting Type // Number
        # Edit to remove mono
        print(self.fit_params.n_components)
        print(self.models)
        self.fit_type = fitting_widgets.ComboBox(
            value=(
                self.models[
                    1 + self.fit_params.n_components - 1
                    ]  # hotfix since n_componentes is 3 but only 2 elenents in list
                if self.fit_params.n_components is not None
                else self.models[0]
            ),
            range_=self.models,
            dtype=int,
            tooltip="Number of Components to fit",
        )
        self.fit_type.currentIndexChanged.connect(self._fit_type_changed)
        self.add_parameter("Fitting Type:", self.fit_type)

        self._init_boundaries()


class NNLSParameterLayout(ParameterLayout):
    def __init__(self, parent: FittingDlg):
        super().__init__(parent)
        self.parent.setWindowTitle("Fitting: NNLS")
        self.reg_order_list = ["0", "1", "2", "3", "CV"]
        self._init_advanced_parameters()
        self.refresh_layout()

    def _init_advanced_parameters(self):
        self.add_seperator()

        # Fitting Type // Regularisation Order
        self.reg_order = fitting_widgets.ComboBox(
            value=str(self.parent.fit_params.reg_order),
            range_=self.reg_order_list,
            dtype=int,
            tooltip="Regularisation Order or Cross Validation Approach",
        )
        self.reg_order.currentIndexChanged.connect(self._reg_order_changed)
        self.add_parameter("Regularisation Order:", self.reg_order)

        # Number of Bins
        self.n_bins = fitting_widgets.EditField(
            value=self.parent.fit_params.boundaries["n_bins"],
            range_=[0, np.power(10, 6)],
            dtype=int,
            tooltip="Number of bins (Diffusion Components) to use for fitting",
        )
        self.add_parameter("Number of Bins:", self.n_bins)

        # Diffusion Range
        self.d_range = fitting_widgets.EditField(
            value=self.parent.fit_params.boundaries["d_range"],
            range_=[0, 1],
            dtype=np.ndarray,
            tooltip="Diffusion Range to place bins in",
        )
        self.add_parameter("Diffusion Range:", self.d_range)

        # Regularisation Factor mu
        self.reg_factor = fitting_widgets.EditField(
            value=self.parent.fit_params.mu,
            range_=[0.0, 1.0],
            dtype=float,
            tooltip="Regularisation factor mu for different Regularisation Orders. \nNot for Cross Validation Approach.",
        )
        self.add_parameter("Regularisation Factor:", self.reg_factor)

        # Cross Validation Tolerance
        self.reg_cv_tol = fitting_widgets.EditField(
            value=(
                getattr(self.parent.fit_params, "tol")
                if hasattr(self.parent.fit_params, "tol")
                else None
            ),
            range_=[0.0, 1.0],
            dtype=float,
            tooltip="Tolerance for Cross Validation Regularisation",
        )

        self.add_parameter("CV Tolerance:", self.reg_cv_tol)

    def refresh_layout(self):
        """Refresh UI elements and activate elements accordingly."""
        # super().refresh_ui()
        if isinstance(self.parent.fit_params, params.NNLSParams):
            self.reg_cv_tol.setEnabled(False)
            self.reg_factor.setEnabled(False)
        elif isinstance(self.parent.fit_params, params.NNLSregParams):
            self.reg_cv_tol.setEnabled(False)
            self.reg_factor.setEnabled(True)
        elif isinstance(self.parent.fit_params, params.NNLSregCVParams):
            self.reg_cv_tol.setEnabled(True)
            self.reg_factor.setEnabled(False)

    def get_parameters(self):
        super().get_parameters()
        self.parent.fit_params.reg_order = self.reg_order.value
        self.parent.fit_params.n_bins = self.n_bins.value
        self.parent.fit_params.d_range = self.d_range.value
        if isinstance(self.parent.fit_params, params.NNLSregParams):
            self.parent.fit_params.reg_factor = self.reg_factor.value
            self.parent.fit_params.mu = self.reg_factor.value
        if isinstance(self.parent.fit_params, params.NNLSregCVParams):
            self.parent.fit_params.reg_factor = self.reg_factor.value
            self.parent.fit_params.tol = self.reg_cv_tol
        return self.parent.fit_params

    def _reg_order_changed(self):
        if self.reg_order.currentText() == self.reg_order_list[0]:
            self.fit_params = params.NNLSParams(
                Path(r"resources/fitting/default_params_NNLS.json")
            )
        elif self.reg_order.currentText() in self.reg_order_list[1:4]:
            self.fit_params = params.NNLSregParams(
                Path(r"resources/fitting/default_params_NNLSreg.json")
            )
        elif self.reg_order.currentText() == self.reg_order_list[4]:
            self.fit_params = params.NNLSregCVParams(
                Path(r"resources/fitting/default_params_NNLSregCV.json")
            )

        if isinstance(self.fit_params, params.NNLSregParams):
            self.reg_factor.value = self.fit_params.mu
        elif isinstance(self.fit_params, params.NNLSregCVParams):
            self.reg_cv_tol.value = self.fit_params.tol

        # self.refresh_ui()


class AcceptButtonLayout(QtWidgets.QHBoxLayout):
    def __init__(self, parent: FittingDlg):
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
        self.button = QtWidgets.QPushButton()
        self.button.setText("Run")
        self.button.setMaximumWidth(75)
        self.button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.button.setIcon(
            self.parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
            )
        )
        self.button.setMaximumHeight(self.height)
        self.addWidget(self.button)
        self.button.clicked.connect(self.accept)
        # TODO: Focus is not working as intended (PopOs at least)
        # self.button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        # self.button.setFocus()
        self.button.isDefault = True

    def accept(self):
        """Accept button callback"""
        # self.output_dict = dict()
        # for key in self.parent.fit_dict:
        #     self.parent.fit_dict[key].current_value = self.parent.fit_dict[key].value
        self.parent.run = True
        self.parent.close()
        self.parent.accept()


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

    menu_bar: FittingMenuBar
    accept_button: AcceptButtonLayout
    main_layout: QVBoxLayout
    parameters: ParameterLayout

    def __init__(self, parent: MainWindow,
                 fit_params: params.Parameters | params.IVIMParams | params.IDEALParams | params.NNLSParams | params.NNLSregParams | params.NNLSregCVParams):
        """Main witting DLG window."""
        super().__init__()

        self.parent = parent
        self.main_window = parent
        self.data = parent.data  # needed?
        self.fit_params = fit_params
        self.setup_ui()
        self.setup_main_layout()

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

    def setup_main_layout(self):
        # Add MainLayout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        # init Menubar
        self.menu_bar = FittingMenuBar(self)
        self.main_layout.addLayout(self.menu_bar)

        # init Parameter Layout
        self.setup_advanced_parameters()
        self.main_layout.addWidget(SeperatorWidget())

        # Add Accept Button Layout
        self.accept_button = AcceptButtonLayout(self)

    def setup_advanced_parameters(self):
        # if isinstance(self.fit_params, params.Parameters):
        #     self.parameters = ParameterLayout(self)
        if isinstance(self.fit_params, params.IVIMParams):
            self.parameters = IVIMParameterLayout(self)
        elif isinstance(self.fit_params, (params.NNLSParams, params.NNLSregParams, params.NNLSregCVParams)):
            self.parameters = NNLSParameterLayout(self)
        elif isinstance(self.fit_params, params.IVIMParams):
            self.parameters = IDEALParameterLayout(self)
        else:
            self.parameters = ParameterLayout(self)
        self.main_layout.addLayout(self.parameters)

    def showEvent(self, event):
        # Add Accept as last element of the dialog main layout
        # This might be obsolet due to dlg to layout change

        self.main_layout.addLayout(self.accept_button)
        self.accept_button.button.setFocus()

    # @abstractmethod
    # def _init_advanced_parameters(self):
    #     """Load Fit specific parameters."""
    #     pass

    # def add_bottom_layout(self):
    #     """
    #     Adds accept button layout to the bottom of the Dialog.
    #
    #     Needs to be called after inheritance to ensure "tab" order.
    #     """

    # def refresh_ui(self):
    #     """Refresh UI elements."""
    #     # self.accept_button.button.setFocus()
    #     pass

    # @abstractmethod
    # def get_parameters(self) -> params.Parameters:
    #     self.fit_params.fit_area = self.parameters.fit_area.value
    #     self.fit_params.scale_image = self.parameters.scale_image.value
    #     self.fit_params.b_values = self.parameters.b_values.value
    #     self.fit_params.max_iter = self.parameters.max_iterations.value
    #     return self.fit_params

# class IVIMFittingDlg(FittingDlg):
#     upper_boundaries: fitting_widgets.EditField
#     lower_boundaries: fitting_widgets.EditField
#     start_values: fitting_widgets.EditField
#     fit_type: fitting_widgets.ComboBox
#
#     def __init__(self, parent: MainWindow, fit_params: params.IVIMParams):
#         super().__init__(parent, fit_params)
#         self.fit_params = fit_params
#         self.setWindowTitle("Fitting: IVIM")
#         self.models = ["MonoExp", "BiExp", "TriExp"]
#         self._init_advanced_parameters()
#         self.refresh_ui()
#
#     @abstractmethod
#     def _init_advanced_parameters(self):
#         self.parameters.add_seperator()
#
#         # Fitting Type // Number
#         # TODO: Number of components needs further refinement
#         self.fit_type = fitting_widgets.ComboBox(
#             value=(
#                 self.models[self.fit_params.n_components - 1]
#                 if self.fit_params.n_components is not None
#                 else self.models[0]
#             ),
#             range_=self.models,
#             dtype=int,
#             tooltip="Number of Components to fit",
#         )
#         self.fit_type.currentIndexChanged.connect(self._fit_type_changed)
#         self.parameters.add_parameter("Fitting Type:", self.fit_type)
#
#         self._init_boundaries()
#
#     def _init_boundaries(self):
#         # X0
#         self.start_values = fitting_widgets.EditField(
#             value=self.fit_params.boundaries["x0"],
#             range_=None,
#             dtype=np.ndarray,
#             tooltip="Start Values",
#         )
#         self.parameters.add_parameter("Start Values:", self.start_values)
#
#         # lb
#         self.lower_boundaries = fitting_widgets.EditField(
#             value=self.fit_params.boundaries["lb"],
#             range_=None,
#             dtype=np.ndarray,
#             tooltip="Lower fitting Boundaries",
#         )
#         self.parameters.add_parameter("Lower Boundaries:", self.lower_boundaries)
#
#         # ub
#         self.upper_boundaries = fitting_widgets.EditField(
#             value=self.fit_params.boundaries["ub"],
#             range_=None,
#             dtype=np.ndarray,
#             tooltip="Upper fitting Boundaries",
#         )
#         self.parameters.add_parameter("Upper Boundaries:", self.upper_boundaries)
#
#     def get_parameters(self) -> params.IVIMParams:
#         super().get_parameters()
#         self.fit_params.boundaries["x0"] = self.start_values.value
#         self.fit_params.boundaries["lb"] = self.lower_boundaries.value
#         self.fit_params.boundaries["ub"] = self.upper_boundaries.value
#         return self.fit_params
#
#     def _fit_type_changed(self):
#         if self.fit_type.currentText() == self.models[0]:
#             self.fit_params = params.IVIMParams(
#                 Path(r"resources/fitting/default_params_IVIM_mono.json")
#             )
#         elif self.fit_type.currentText() == self.models[1]:
#             self.fit_params = params.IVIMParams(
#                 Path(r"resources/fitting/default_params_IVIM_bi.json")
#             )
#         elif self.fit_type.currentText() == self.models[2]:
#             self.fit_params = params.IVIMParams(
#                 Path(r"resources/fitting/default_params_IVIM_tri.json")
#             )
#         else:
#             print("Selected model didn't fit to any listed Models.")
#             return
#
#         self.start_values.value = self.fit_params.boundaries["x0"]
#         self.lower_boundaries.value = self.fit_params.boundaries["lb"]
#         self.upper_boundaries.value = self.fit_params.boundaries["ub"]
#
#         self.refresh_ui()
#
#     def refresh_ui(self):
#         super().refresh_ui()


# class IDEALFittingDlg(IVIMFittingDlg):
#     def __init__(self, parent: MainWindow, fit_params: params.IDEALParams):
#         super().__init__(parent, fit_params)
#         self.models = ["BiExp", "TriExp"]
#         self._init_advanced_parameters()
#
#     def get_parameters(self):
#         pass
#
#     def _fit_type_changed(self):
#         if self.fit_type.currentText() == self.models[0]:
#             self.fit_params = params.IVIMParams(
#                 Path(r"resources/fitting/default_params_ideal_bi.json")
#             )
#         elif self.fit_type.currentText() == self.models[1]:
#             self.fit_params = params.IVIMParams(
#                 Path(r"resources/fitting/default_params_ideal_tri.json")
#             )
#         else:
#             print("Selected model didn't fit to any listed Models.")
#             return
#
#         self.start_values.value = self.fit_params.boundaries["x0"]
#         self.lower_boundaries.value = self.fit_params.boundaries["lb"]
#         self.upper_boundaries.value = self.fit_params.boundaries["ub"]
#
#         self.refresh_ui()
#
#     def _init_advanced_parameters(self):
#         # super()._init_advanced_parameters()
#         self.parameters.add_seperator()
#
#         # Fitting Type // Number
#         print(self.fit_params.n_components)
#         print(self.models)
#         self.fit_type = fitting_widgets.ComboBox(
#             value=(
#                 self.models[
#                     1 + self.fit_params.n_components - 1
#                     ]  # hotfix since n_componentes is 3 but only 2 elenents in list
#                 if self.fit_params.n_components is not None
#                 else self.models[0]
#             ),
#             range_=self.models,
#             dtype=int,
#             tooltip="Number of Components to fit",
#         )
#         self.fit_type.currentIndexChanged.connect(self._fit_type_changed)
#         self.parameters.add_parameter("Fitting Type:", self.fit_type)
#
#         self._init_boundaries()


# class NNLSFittingDlg(FittingDlg):
#     def __init__(
#         self,
#         parent: MainWindow,
#         fit_params: params.NNLSParams | params.NNLSregParams | params.NNLSregCVParams,
#     ):
#         super().__init__(parent, fit_params)
#         self.fit_params = fit_params
#         self.reg_order_list = ["0", "1", "2", "3", "CV"]
#         self.setWindowTitle("Fitting: NNLS")
#         self._init_advanced_parameters()
#         self.refresh_ui()
#
#     def _init_advanced_parameters(self):
#         self.parameters.add_seperator()
#
#         # Fitting Type // Regularisation Order
#         self.reg_order = fitting_widgets.ComboBox(
#             value=str(self.fit_params.reg_order),
#             range_=self.reg_order_list,
#             dtype=int,
#             tooltip="Regularisation Order or Cross Validation Approach",
#         )
#         self.reg_order.currentIndexChanged.connect(self._reg_order_changed)
#         self.parameters.add_parameter("Regularisation Order:", self.reg_order)
#
#         # Number of Bins
#         self.n_bins = fitting_widgets.EditField(
#             value=self.fit_params.boundaries["n_bins"],
#             range_=[0, np.power(10, 6)],
#             dtype=int,
#             tooltip="Number of bins (Diffusion Components) to use for fitting",
#         )
#         self.parameters.add_parameter("Number of Bins:", self.n_bins)
#
#         # Diffusion Range
#         self.d_range = fitting_widgets.EditField(
#             value=self.fit_params.boundaries["d_range"],
#             range_=[0, 1],
#             dtype=np.ndarray,
#             tooltip="Diffusion Range to place bins in",
#         )
#         self.parameters.add_parameter("Diffusion Range:", self.d_range)
#
#         # Regularisation Factor mu
#         self.reg_factor = fitting_widgets.EditField(
#             value=self.fit_params.mu,
#             range_=[0.0, 1.0],
#             dtype=float,
#             tooltip="Regularisation factor mu for different Regularisation Orders. \nNot for Cross Validation Approach.",
#         )
#         self.parameters.add_parameter("Regularisation Factor:", self.reg_factor)
#
#         # Cross Validation Tolerance
#         self.reg_cv_tol = fitting_widgets.EditField(
#             value=(
#                 getattr(self.fit_params, "tol")
#                 if hasattr(self.fit_params, "tol")
#                 else None
#             ),
#             range_=[0.0, 1.0],
#             dtype=float,
#             tooltip="Tolerance for Cross Validation Regularisation",
#         )
#
#         self.parameters.add_parameter("CV Tolerance:", self.reg_cv_tol)
#
#     def refresh_ui(self):
#         """Refresh UI elements and activate elements accordingly."""
#         super().refresh_ui()
#         if isinstance(self.fit_params, params.NNLSregParams):
#             self.reg_cv_tol.setEnabled(False)
#             self.reg_factor.setEnabled(True)
#         elif isinstance(self.fit_params, params.NNLSregCVParams):
#             self.reg_cv_tol.setEnabled(True)
#             self.reg_factor.setEnabled(False)
#
#     def get_parameters(self):
#         super().get_parameters()
#         self.fit_params.reg_order = self.reg_order.value
#         self.fit_params.n_bins = self.n_bins.value
#         self.fit_params.d_range = self.d_range.value
#         if isinstance(self.fit_params, params.NNLSregParams):
#             self.fit_params.reg_factor = self.reg_factor.value
#             self.fit_params.mu = self.reg_factor.value
#         if isinstance(self.fit_params, params.NNLSregCVParams):
#             self.fit_params.reg_factor = self.reg_factor.value
#             self.fit_params.tol = self.reg_cv_tol
#         return self.fit_params
#
#     def _reg_order_changed(self):
#         if self.reg_order.currentText() == self.reg_order_list[0]:
#             self.fit_params = params.NNLSParams(
#                 Path(r"resources/fitting/default_params_NNLS.json")
#             )
#         elif self.reg_order.currentText() in self.reg_order_list[1:4]:
#             self.fit_params = params.NNLSregParams(
#                 Path(r"resources/fitting/default_params_NNLSreg.json")
#             )
#         elif self.reg_order.currentText() == self.reg_order_list[4]:
#             self.fit_params = params.NNLSregCVParams(
#                 Path(r"resources/fitting/default_params_NNLSregCV.json")
#             )
#
#         if isinstance(self.fit_params, params.NNLSregParams):
#             self.reg_factor.value = self.fit_params.mu
#         elif isinstance(self.fit_params, params.NNLSregCVParams):
#             self.reg_cv_tol.value = self.fit_params.tol
#
#         self.refresh_ui()
