from __future__ import annotations
import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtGui
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QVBoxLayout

from . import widgets_fitting as fitting_widgets
from ..utils.exceptions import ClassMismatch
from ..fit import parameters as params

if TYPE_CHECKING:
    from PyNeapple_UI import MainWindow


class SeperatorWidget(QtWidgets.QFrame):
    """Returns a horizontal separator for list style Layout."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


class FittingMenuBar(QtWidgets.QVBoxLayout):
    """Initialize the fitting menu bar for fitting dialogues."""

    def __init__(self, parent: FittingDlg):
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
                self.parent.parameters.set_parameters()

                # TODO: Let loaded Parameters change layout currently unable to remove old widgets
                # self.parent.parameters.unload_parameters()
                # self.parent.parameters.deleteLater()
                # self.parent.fit_params = params.JsonImporter(path).load_json()
                # if isinstance(self.parent.fit_params, params.IVIMParams):
                #     self.parent.parameters = IVIMParameterLayout(self.parent)
                # elif isinstance(
                #     self.parent.fit_params,
                #     (params.NNLSParams, params.NNLSCVParams),
                # ):
                #     self.parent.parameters = NNLSParameterLayout(self.parent)
                # elif isinstance(self.parent.fit_params, params.IDEALParams):
                #     self.parent.parameters = IDEALParameterLayout(self.parent)
                # else:
                #     self.parent.parameters = ParameterLayout(self.parent)
            except ClassMismatch:
                print("Warning: No supported Parameter Class detected!")
                return None

            # self.parent.parameters = parameters
            # self.parent.fit_params = fit_params
            # self.parent.main_layout.addLayout(self.parent.parameters)
            # self.parent.main_layout.addLayout(self.parent.accept_button)
            self.parent.parent.data.last_dir = path.parent

    def save_action_pushed(self):
        """Save fit parameters to json file."""
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
    """Layout holding general fitting parameters."""

    def __init__(self, parent: FittingDlg):
        super().__init__()
        self.parent = parent
        self.row_iterator = 0

        self.add_seperator()

        # FitArea
        self.fit_area = fitting_widgets.ComboBox(
            value=self.parent.fit_params.fit_area,
            range_=["Pixel", "Segmentation"],
            dtype=str,
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
        """Add parameter Widget with label to Grid."""
        self.addWidget(QtWidgets.QLabel(text), self.row_iterator, 0)
        if kwargs.get("alignment", None):
            self.addWidget(
                widget, self.row_iterator, 1, alignment=kwargs.get("alignment")
            )
        else:
            self.addWidget(widget, self.row_iterator, 1)
        self.row_iterator += 1

    def add_seperator(self):
        """Add seperator to Grid Layout."""
        self.addWidget(SeperatorWidget(), self.row_iterator, 0, 1, 2)
        self.row_iterator += 1

    # def unload_parameters(self):
    #     # for i in reversed(range(self.count())):
    #     #     self.removeItem(self.itemAt(i))
    #     # self.deleteLater()
    #     self.removeWidget(self.fit_area)
    #     self.removeWidget(self.scale_image)
    #     self.removeWidget(self.b_values)
    #     self.removeWidget(self.max_iterations)

    def _load_b_values(self):
        """Callback to load b-values from file."""
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
        """Get parameters from Widgets."""
        self.parent.fit_params.fit_area = self.fit_area.value
        if self.scale_image.value:
            self.parent.fit_params.scale_image = "S/S0"
        else:
            self.parent.fit_params.scale_image = self.scale_image.value
        self.parent.fit_params.b_values = self.b_values.value
        self.parent.fit_params.max_iter = self.max_iterations.value
        return self.parent.fit_params

    def set_parameters(self):
        """Set parameters from class to Widgets"""
        self.fit_area.value = self.parent.fit_params.fit_area
        self.scale_image.value = self.parent.fit_params.scale_image
        self.b_values.value = self.parent.fit_params.b_values
        self.max_iterations.value = self.parent.fit_params.max_iter


class IVIMParameterLayout(ParameterLayout):
    """Layout holding IVIM fitting parameters."""

    def __init__(self, parent: FittingDlg):
        super().__init__(parent)
        self.parent.setWindowTitle("Fitting: IVIM")
        self._init_advanced_parameters()

    def _init_advanced_parameters(self):
        """Load advanced fitting parameter widgets for IVIM."""
        self.models = ["MonoExp", "BiExp", "TriExp"]
        self.add_seperator()

        # Fitting Type // Number
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
        """Add boundary widgets."""
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

    def _fit_type_changed(self):
        """Callback for fit type ComboBox."""
        if self.fit_type.currentText() == self.models[0]:
            self.parent.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_IVIM_mono.json")
            )
        elif self.fit_type.currentText() == self.models[1]:
            self.parent.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_IVIM_bi.json")
            )
        elif self.fit_type.currentText() == self.models[2]:
            self.parent.fit_params = params.IVIMParams(
                Path(r"resources/fitting/default_params_IVIM_tri.json")
            )
        else:
            print("Selected model didn't fit to any listed Models.")
            return

        self.start_values.value = self.parent.fit_params.boundaries["x0"]
        self.lower_boundaries.value = self.parent.fit_params.boundaries["lb"]
        self.upper_boundaries.value = self.parent.fit_params.boundaries["ub"]

    def get_parameters(self) -> params.IVIMParams:
        """Get parameters from Widgets."""
        super().get_parameters()
        self.parent.fit_params.boundaries["x0"] = self.start_values.value
        self.parent.fit_params.boundaries["lb"] = self.lower_boundaries.value
        self.parent.fit_params.boundaries["ub"] = self.upper_boundaries.value
        return self.parent.fit_params

    def set_parameters(self):
        """Set parameters from class to Widgets"""
        super().set_parameters()
        self.start_values.value = self.parent.fit_params.boundaries["x0"]
        self.lower_boundaries.value = self.parent.fit_params.boundaries["lb"]
        self.upper_boundaries.value = self.parent.fit_params.boundaries["ub"]

    # def unload_parameters(self):
    #     super().unload_parameters()


class IDEALParameterLayout(IVIMParameterLayout):
    """Layout holding IDEAL fitting parameters."""

    def __init__(self, parent: FittingDlg):
        super().__init__(parent)
        self.parent.setWindowTitle("Fitting: IDEAL")
        # self.models = ["BiExp", "TriExp"]
        # self._init_advanced_parameters()
        self.fit_area.setEnabled(False)

    # def unload_parameters(self):
    #     super().unload_parameters()

    def _fit_type_changed(self):
        """Callback for fit type change."""
        if self.fit_type.currentText() == self.models[0]:
            self.parent.fit_params = params.IDEALParams(
                Path(r"resources/fitting/default_params_IDEAL_bi.json")
            )
        elif self.fit_type.currentText() == self.models[1]:
            self.parent.fit_params = params.IDEALParams(
                Path(r"resources/fitting/default_params_IDEAL_tri.json")
            )
        else:
            print("Selected model didn't fit to any listed Models.")
            return

        self.start_values.value = self.parent.fit_params.boundaries["x0"]
        self.lower_boundaries.value = self.parent.fit_params.boundaries["lb"]
        self.upper_boundaries.value = self.parent.fit_params.boundaries["ub"]

        # self.refresh_ui()

    def _init_advanced_parameters(self):
        """Load advanced fitting parameter widgets for IDEAL."""

        self.models = ["BiExp", "TriExp"]
        self.add_seperator()

        # Fitting Type // Number
        # Edit to remove mono
        self.fit_type = fitting_widgets.ComboBox(
            value=(
                self.models[
                    self.parent.fit_params.n_components - 2
                ]  # hotfix since n_componentes is 3 but only 2 elements in list
                if self.parent.fit_params.n_components is not None
                and self.parent.fit_params.n_components
                > 1  # take removed mono into account
                else self.models[0]
            ),
            range_=self.models,
            dtype=int,
            tooltip="Number of Components to fit",
        )
        self.fit_type.currentIndexChanged.connect(self._fit_type_changed)
        self.add_parameter("Fitting Type:", self.fit_type)

        self._init_boundaries()

    def get_parameters(self):
        """Get parameters from Widgets."""
        return super().get_parameters()

    def set_parameters(self):
        """Set parameters from class to Widgets."""
        return super().set_parameters()


class NNLSParameterLayout(ParameterLayout):
    """Layout holding NNLS fitting parameters."""

    def __init__(self, parent: FittingDlg):
        super().__init__(parent)
        self.parent.setWindowTitle("Fitting: NNLS")
        self.reg_order_list = ["0", "1", "2", "3", "CV"]
        self._init_advanced_parameters()
        self._refresh_layout()

    def _init_advanced_parameters(self):
        """Add NNLS specific parameters to the layout."""
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
            value=(
                getattr(self.parent.fit_params, "mu")
                if hasattr(self.parent.fit_params, "mu")
                else None
            ),
            range_=[0.0, 1.0],
            dtype=float,
            tooltip="Regularisation factor mu for different Regularisation Orders.\nNot for Cross Validation Approach.",
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

    # def unload_parameters(self):
    #     super().unload_parameters()

    def _reg_order_changed(self):
        """Callback for changes of the reg order combobox."""

        if self.reg_order.currentText() in self.reg_order_list[0:4]:
            self.parent.fit_params = params.NNLSParams(
                Path(r"resources/fitting/default_params_NNLS.json")
            )
            if self.reg_factor.value is not None:
                self.reg_factor.value = self.parent.fit_params.mu
        elif self.reg_order.currentText() == self.reg_order_list[4]:
            self.parent.fit_params = params.NNLSCVParams(
                Path(r"resources/fitting/default_params_NNLSCV.json")
            )
            if self.reg_cv_tol.value is not None:
                self.reg_cv_tol.value = self.parent.fit_params.tol

        self._refresh_layout()

    def _refresh_layout(self):
        """Refresh UI elements and activate elements accordingly."""
        # super().refresh_ui()

        if isinstance(self.parent.fit_params, params.NNLSParams):
            self.reg_cv_tol.setEnabled(False)
            self.reg_factor.setEnabled(True)
        elif isinstance(self.parent.fit_params, params.NNLSCVParams):
            self.reg_cv_tol.setEnabled(True)
            self.reg_factor.setEnabled(False)
        else:  # if isinstance(self.parent.fit_params, params.NNLSbaseParams):
            self.reg_cv_tol.setEnabled(False)
            self.reg_factor.setEnabled(False)

    def get_parameters(self):
        """Get parameters from Widgets."""
        super().get_parameters()
        self.parent.fit_params.reg_order = self.reg_order.value
        self.parent.fit_params.n_bins = self.n_bins.value
        self.parent.fit_params.d_range = self.d_range.value
        if isinstance(self.parent.fit_params, params.NNLSParams):
            self.parent.fit_params.mu = self.reg_factor.value
        if isinstance(self.parent.fit_params, params.NNLSCVParams):
            self.parent.fit_params.tol = self.reg_cv_tol.value
        return self.parent.fit_params

    def set_parameters(self):
        """Set parameters from class to Widgets."""
        super().set_parameters()
        self.reg_order.value = self.parent.fit_params.reg_order
        self.n_bins.value = self.parent.fit_params.boundaries["n_bins"]
        self.d_range.value = self.parent.fit_params.boundaries["d_range"]
        if isinstance(self.parent.fit_params, params.NNLSParams):
            self.reg_factor.value = self.parent.fit_params.mu
        if isinstance(self.parent.fit_params, params.NNLSCVParams):
            self.reg_cv_tol.value = self.parent.fit_params.tol


class AcceptButtonLayout(QtWidgets.QHBoxLayout):
    def __init__(self, parent: FittingDlg):
        """Layout for accept button at the bottom of the dialog"""
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

    def __init__(
        self,
        parent: MainWindow,
        fit_params: (
            params.Parameters
            | params.IVIMParams
            | params.IDEALParams
            | params.NNLSParams
            | params.NNLSCVParams
        ),
    ):
        """Main witting DLG window."""
        super().__init__()

        self.parent = parent
        self.main_window = parent
        self.data = parent.data  # needed?
        self.fit_params = fit_params
        self.setup_ui()
        self.setup_main_layout()

    def setup_ui(self):
        """Setup UI elements."""
        self.setWindowTitle("Fitting")
        self.setWindowIcon(
            QtGui.QIcon(
                Path(
                    Path(__file__).parent.parent.parent.parent,
                    "resources",
                    "images",
                    "PyNeappleLogo.ico",
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
        """Setup main layout for parameters and other UI elements."""
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
        """Setup advanced fit specific parameters."""
        if isinstance(self.fit_params, params.IDEALParams):
            self.parameters = IDEALParameterLayout(self)
        elif isinstance(
            self.fit_params,
            (params.NNLSParams, params.NNLSCVParams),
        ):
            self.parameters = NNLSParameterLayout(self)
        elif isinstance(self.fit_params, params.IVIMParams):
            self.parameters = IVIMParameterLayout(self)
        else:
            self.parameters = ParameterLayout(self)
        self.main_layout.addLayout(self.parameters)

    def showEvent(self, event):
        """Show the dialog event."""
        # Add Accept as last element of the dialog main layout
        # This might be obsolete due to dlg to layout change

        self.main_layout.addLayout(self.accept_button)
        self.accept_button.button.setFocus()
