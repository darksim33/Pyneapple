from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction  # , QIcon

from src.utils import Nii, NiiSeg
from src.ui.dialogues.prompt_dlg import (
    FitParametersDlg,
    MissingSegDlg,
    IDEALDimensionDlg,
)
from src.ui.dialogues.fitting_dlg import FittingDlg, FittingDictionaries
from src.fit import parameters

if TYPE_CHECKING:
    from PyNeapple_UI import MainWindow


class FitAction(QAction):
    def __init__(
        self,
        parent: MainWindow,
        text: str,
        model_name: str,
        # icon: QIcon | None = None,
    ):
        """
        Basic Class to set up fitting Action for different Algorithms.

        The main fit function is currently deployed here.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the action
            text: str
                Set the text of the menu item
            model_name: str
                Identify the model that will be used to fit the data
            # icon: QIcon | None
                Set the icon of the action

                Create a new instance of the class
        """
        super().__init__(parent=parent, text=text)
        self.parent = parent
        self.model_name = model_name
        self.triggered.connect(self.setup_fit)

    @property
    def fit_data(self):
        return self.parent.data.fit_data

    @abstractmethod
    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        pass

    @abstractmethod
    def get_dlg_dict(self) -> dict:
        """Return dictionary for fitting dialog."""
        pass

    @abstractmethod
    def load_parameters_from_dlg_dict(self):
        """Set Fit specific parameters."""
        b_values = self.b_values_from_dict()
        self.fit_data.fit_params.b_values = b_values
        self.fit_data.fit_params.n_pools = self.parent.settings.value(
            "number_of_pools", type=int
        )
        self.parent.fit_dlg.dict_to_attributes(self.fit_data.fit_params)

    @abstractmethod
    def check_fit_parameters(self):
        pass

    def fit_run(self):
        if self.fit_data.fit_params.fit_area == "Pixel":
            self.fit_data.fit_pixel_wise(
                multi_threading=self.parent.settings.value("multithreading", type=bool)
            )
        elif self.fit_data.fit_area == "Segmentation":
            self.fit_data.fit_segmentation_wise()

    def b_values_from_dict(self):
        """
        Extract b_values from the fit dialog.

        The b_values_from_dict function is used to extract the b_values from the fit_dict.
        The function first checks if there are any b_values in the fit dict, and if so, it extracts them.
        If they are a string, then it converts them into an array of integers using numpy's fromstring method.
        It then reshapes this array to match that of self.parent.data.fit_data (the data object).
        If they were not a string but instead a list or some other type of iterable object, then we simply convert them
        into an array using numpy's nparray method.
        """
        b_values = self.parent.fit_dlg.fit_dict.pop("b_values", False).value
        if b_values:
            if isinstance(b_values, str):
                b_values = np.fromstring(
                    b_values.replace("[", "").replace("]", ""), dtype=int, sep="  "
                )
                if (
                    b_values.shape
                    != self.parent.data.fit_data.fit_params.b_values.shape
                ):
                    b_values = np.reshape(
                        b_values, self.parent.data.fit_data.fit_params.b_values.shape
                    )
            elif isinstance(b_values, list):
                b_values = np.array(b_values)

            return b_values
        else:
            return self.parent.data.fit_data.fit_params.b_values

    def setup_fit(self):
        """
        Main pyneapple fitting function for the UI.

        Handles IVIM, IDEAL and NNLS fitting.
        """

        # Validate current parameters
        self.set_parameter_instance()

        # Load dict for dialog
        dlg_dict = self.get_dlg_dict()

        # Launch Dlg
        self.parent.fit_dlg = FittingDlg(
            self.model_name,
            dlg_dict,
            self.fit_data.fit_params,
            app_data=self.parent.data,
        )
        self.parent.fit_dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.parent.fit_dlg.exec()
        # Load parameters from dialog
        self.load_parameters_from_dlg_dict()

        # Prepare Data
        self.fit_data.img = self.parent.data.nii_img
        self.fit_data.seg = self.parent.data.nii_seg

        if self.parent.fit_dlg.run:
            self.parent.mainWidget.setCursor(QtCore.Qt.CursorShape.WaitCursor)

            self.check_fit_parameters()
            # Check if seg is present else create new one
            if not self.fit_data.seg.path:
                missing_seg_dlg = MissingSegDlg()
                # result = missing_seg_dlg.exec()
                if missing_seg_dlg.exec():
                    array = np.ones(self.fit_data.img.array.shape)
                    self.fit_data.seg = self.parent.data.nii_seg = NiiSeg().from_array(
                        np.expand_dims(array[:, :, :, 1], 3)
                    )

            self.fit_run()
            self.parent.data.nii_dyn = Nii().from_array(
                self.parent.data.fit_data.fit_results.spectrum
            )


            # Save fit results into dynamic nii struct for plotting the spectrum
            self.parent.data.nii_dyn = Nii().from_array(
                self.parent.data.fit_data.fit_results.spectrum
            )

            self.parent.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

            self.parent.file_menu.save_fit_image.setEnabled(True)


class NNLSFitAction(FitAction):
    def __init__(self, parent: MainWindow):
        """NNLS Fit Action."""
        super().__init__(parent=parent, text="NNLS...", model_name="NNLS")

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(
            self.fit_data.fit_params,
            (
                parameters.NNLSParams
                or parameters.NNLSregParams
                or parameters.NNLSregCVParams
            ),
        ):
            if isinstance(self.fit_data.fit_params, parameters.Parameters):
                self.fit_data.fit_params = parameters.NNLSregParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_NNLSreg.json",
                    )
                )
            else:
                dialog = FitParametersDlg(self.fit_data.fit_params)
                result = dialog.exec()
                if result:
                    self.fit_data.fit_params = parameters.NNLSregParams(
                        Path(
                            self.parent.data.app_path,
                            "resources",
                            "fitting",
                            "default_params_NNLSreg.json",
                        )
                    )
                else:
                    return
        self.fit_data.model_name = "NNLS"

    def get_dlg_dict(self) -> dict:
        """ "Return dictionary for fitting dialog."""
        return FittingDictionaries.get_nnls_dict(self.fit_data.fit_params)

    def load_parameters_from_dlg_dict(self):
        """Load Parameters from Dialog Dictionary."""
        super().load_parameters_from_dlg_dict()
        if self.fit_data.fit_params.reg_order == "CV":
            self.fit_data.fit_params = parameters.NNLSregCVParams()
            self.parent.fit_dlg.dict_to_attributes(self.fit_data.fit_params)
        elif self.fit_data.fit_params.reg_order != "CV":
            self.fit_data.fit_params.reg_order = int(self.fit_data.fit_params.reg_order)
            if self.fit_data.fit_params.reg_order == 0:
                self.fit_data.fit_params = parameters.NNLSParams()
                self.parent.fit_dlg.dict_to_attributes(self.fit_data.fit_params)
        super().load_parameters_from_dlg_dict()

    def check_fit_parameters(self):
        pass


class IVIMFitAction(FitAction):
    def __init__(self, parent: MainWindow):
        """IVIM Fit Action."""
        super().__init__(parent=parent, text="IVIM...", model_name="IVIM")

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(self.fit_data.fit_params, parameters.IVIMParams):
            if isinstance(self.fit_data.fit_params, parameters.Parameters):
                self.fit_data.fit_params = parameters.IVIMParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_IVIM_tri.json",
                    )
                )
            else:
                dialog = FitParametersDlg(self.fit_data.fit_params)
                result = dialog.exec()
                if result:
                    self.fit_data.fit_params = parameters.IVIMParams(
                        Path(
                            self.parent.data.app_path,
                            "resources",
                            "fitting",
                            "default_params_IVIM_tri.json",
                        )
                    )
                else:
                    return None
        self.fit_data.model_name = "IVIM"

    def get_dlg_dict(self) -> dict:
        """Return dictionary for fitting dialog."""
        return FittingDictionaries.get_ivim_dict(self.fit_data.fit_params)

    def load_parameters_from_dlg_dict(self):
        """Load Parameters from Dialog Dictionary."""
        super().load_parameters_from_dlg_dict()

    def check_fit_parameters(self):
        pass


class IDEALFitAction(FitAction):
    def __init__(self, parent: MainWindow):
        """IDEAL IVIM Fit Action"""
        super().__init__(parent=parent, text="IDEAL...", model_name="IDEAL")

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(self.fit_data.fit_params, parameters.IDEALParams):
            if isinstance(self.fit_data.fit_params, parameters.Parameters):
                self.fit_data.fit_params = parameters.IDEALParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_ideal.json",
                    )
                )
            else:
                dialog = FitParametersDlg(self.fit_data.fit_params)
                result = dialog.exec()
                if result:
                    self.fit_data.fit_params = parameters.IDEALParams(
                        Path(
                            self.parent.data.app_path,
                            "resources",
                            "fitting",
                            "default_params_ideal.json",
                        )
                    )
                else:
                    return None
        self.fit_data.model_name = "IDEAL"

    def get_dlg_dict(self) -> dict:
        """Return dictionary for fitting dialog."""
        return FittingDictionaries.get_ideal_dict(self.fit_data.fit_params)

    def load_parameters_from_dlg_dict(self):
        """Load Parameters from Dialog Dictionary."""
        super().load_parameters_from_dlg_dict()

    def check_fit_parameters(self):
        if (
            not self.fit_data.fit_params.dimension_steps[0]
            == self.fit_data.img.array.shape[0:2]
        ):
            print(
                f"Matrix size missmatch! {self.fit_data.fit_params.dimension_steps[0]} vs {self.fit_data.img.array.shape[0:2]}"
            )
            dimension_dlg = IDEALDimensionDlg()
            if dimension_dlg.exec():
                self.fit_data.fit_params.dimension_steps[0] = (
                    self.fit_data.img.array.shape[0:2],
                )

    def fit_run(self):
        self.fit_data.fit_ideal(
            multi_threading=self.parent.settings.value("multithreading", type=bool)
        )


class SaveResultsAction(QAction):
    def __init__(self, parent: MainWindow):
        """Save results to Excel action."""
        super().__init__(
            parent=parent,
            text="Save Results...",
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.parent = parent
        self.triggered.connect(self.save)

    def save(self):
        """Saves results to Excel sheet, saved in dir of img file."""
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Save Results to Excel",
                file.parent.__str__()
                + "\\"
                + file.stem
                + "_"
                + model
                + "_results.xlsx",
                "Excel (*.xlsx)",
            )[0]
        )

        if file_path:
            self.parent.data.fit_data.fit_results.save_results_to_excel(file_path)


class SaveAUCResultsAction(QAction):
    def __init__(self, parent: MainWindow):
        """Save AUC results to Excel action."""
        super().__init__(
            parent=parent,
            text="Save AUC results...",
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.parent = parent
        self.triggered.connect(self.save_AUC)

    def save_AUC(self):
        """Saves results to Excel sheet, saved in dir of img file."""
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Create and save heatmaps",
                file.parent.__str__()
                + "\\"
                + file.stem
                + "_"
                + model
                + "_AUC_results.xlsx",
                "Excel (*.xlsx)",
            )[0]
        )

        if file_path:
            (
                d_AUC,
                f_AUC,
            ) = self.parent.data.fit_data.fit_params.apply_AUC_to_results(
                self.parent.data.fit_data.fit_results
            )
            self.parent.data.fit_data.fit_results.save_results_to_excel(
                file_path, d_AUC, f_AUC
            )


class SaveSpectrumAction(QAction):
    def __init__(self, parent: MainWindow):
        """Save spectrum action."""
        super().__init__(
            parent=parent,
            text="Save Spectrum...",
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.parent = parent
        self.triggered.connect(self.save_spectrum)

    def save_spectrum(self):
        """Saves spectrum as 4D Nii file."""
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Save Spectrum",
                file.parent.__str__() + "\\" + file.stem + "_" + model + "_spec.nii",
                "Nii (*.nii)",
            )[0]
        )

        if file_path:
            self.parent.data.fit_data.fit_results.save_spectrum_to_nii(file_path)


class CreateHeatMapsAction(QAction):
    def __init__(self, parent: MainWindow):
        """Create Heat Maps and save them."""
        super().__init__(
            parent=parent,
            text="Create Heatmaps...",
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.parent = parent
        self.triggered.connect(self.create_heat_maps)

    def create_heat_maps(self):
        """Creates heatmaps for d and f for every slice containing a segmentation."""
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name
        slices_contain_seg = self.parent.data.nii_seg.slices_contain_seg

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Create and save heatmaps",
                file.parent.__str__() + "\\" + file.stem + "_" + model + "_heatmaps",
            )[0]
        )

        if file_path:
            for slice_idx, slice_contains_seg in enumerate(slices_contain_seg):
                if slice_contains_seg:
                    (
                        d_AUC,
                        f_AUC,
                    ) = self.parent.data.fit_data.fit_params.apply_AUC_to_results(
                        self.parent.data.fit_data.fit_results
                    )
                    img_dim = self.parent.data.fit_data.img.array.shape[0:3]

                    self.parent.data.fit_data.fit_results.create_heatmap(
                        img_dim, model, d_AUC, f_AUC, file_path, slice_idx
                    )


class FittingMenu(QMenu):
    fit_NNLS: NNLSFitAction
    fit_IVIM: IVIMFitAction
    fit_IDEAL: IDEALFitAction
    save_results: SaveResultsAction
    save_AUC_results: SaveAUCResultsAction
    save_spectrum: SaveSpectrumAction
    create_heat_maps: CreateHeatMapsAction

    def __init__(self, parent: MainWindow):
        """
        QMenu to handle the basic fitting related actions.

        Parameters
        ----------
            self
                Represent the instance of the class
            parent: MainWindow
                Pass the parent window to the menu
        """
        super().__init__("&Fitting", parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Sets up fitting menu."""
        self.fit_NNLS = NNLSFitAction(self.parent)
        self.addAction(self.fit_NNLS)
        self.fit_IVIM = IVIMFitAction(self.parent)
        self.addAction(self.fit_IVIM)
        self.fit_IDEAL = IDEALFitAction(self.parent)
        self.addAction(self.fit_IDEAL)

        self.addSeparator()
        self.save_results = SaveResultsAction(self.parent)
        self.addAction(self.save_results)
        self.save_AUC_results = SaveAUCResultsAction(self.parent)
        self.addAction(self.save_AUC_results)
        self.save_spectrum = SaveSpectrumAction(self.parent)
        self.addAction(self.save_spectrum)
        self.create_heat_maps = CreateHeatMapsAction(self.parent)
        self.addAction(self.create_heat_maps)
