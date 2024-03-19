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
    FitParametersMessageBox,
    MissingSegmentationMessageBox,
    IDEALDimensionMessageBox,
    RepeatedFitMessageBox,
)
from src.ui.dialogues.fitting_dlg import FittingDlg
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
        # self.parent.data.fit_data = parent.data.fit_data

    @abstractmethod
    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        pass

    @abstractmethod
    def check_fit_parameters(self):
        pass

    def setup_fit(self):
        """
        Main pyneapple fitting function for the UI.

        Handles IVIM, IDEAL and NNLS fitting.
        """
        # check if fit was performed before
        run = self.check_for_previous_fit()
        if not run:
            return

        # Validate current parameters
        self.set_parameter_instance()

        # Launch Dlg
        self.parent.fit_dlg = FittingDlg(
            self.parent, self.parent.data.fit_data.fit_params
        )
        self.parent.fit_dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        run = self.parent.fit_dlg.exec()
        # Load parameters from dialog
        self.parent.data.fit_data.fit_params = (
            self.parent.fit_dlg.parameters.get_parameters()
        )

        # Prepare Data
        # Scale Image if needed
        self.parent.data.fit_data.img = self.parent.data.nii_img.copy()
        self.parent.data.fit_data.img.scale_image(
            self.parent.fit_dlg.fit_params.scale_image
        )
        self.parent.data.fit_data.seg = self.parent.data.nii_seg

        if run:
            # if self.parent.fit_dlg.run:
            self.parent.mainWidget.setCursor(QtCore.Qt.CursorShape.WaitCursor)

            self.check_fit_parameters()
            # Check if seg is present else create new one
            if not self.parent.data.fit_data.seg.path:
                missing_seg_dlg = MissingSegmentationMessageBox()
                if missing_seg_dlg.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                    array = np.ones(self.parent.data.fit_data.img.array.shape)
                    self.parent.data.fit_data.seg = (
                        self.parent.data.nii_seg
                    ) = NiiSeg().from_array(np.expand_dims(array[:, :, :, 1], 3))

            self.fit_run()
            self.parent.data.nii_dyn = Nii().from_array(
                self.parent.data.fit_data.fit_results.spectrum
            )

            # Save fit results into dynamic nii struct for plotting the spectrum
            self.parent.data.nii_dyn = Nii().from_array(
                self.parent.data.fit_data.fit_results.spectrum
            )

            self.parent.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.parent.data.fit_data.flags["did_fit"] = True
            self.parent.file_menu.save_fit_image.setEnabled(True)

    def check_for_previous_fit(self) -> bool:
        if self.parent.data.fit_data.flags.get("did_fit", False):
            print("Warning: There was a Fit performed before.")
            dlg_result = RepeatedFitMessageBox().exec()
            if dlg_result == QtWidgets.QMessageBox.StandardButton.Discard:
                print("Discarding previous Fit...")
                self.parent.data.fit_data.reset()
                self.parent.data.nii_dyn = Nii()
                return True
            elif dlg_result == QtWidgets.QMessageBox.StandardButton.Abort:
                print("Aborting Fit...")
                return False
        else:
            return True

    def fit_run(self):
        if self.parent.data.fit_data.fit_params.fit_area == "Pixel":
            self.parent.data.fit_data.fit_pixel_wise(
                multi_threading=self.parent.settings.value("multithreading", type=bool)
            )
            self.parent.data.plt["plt_type"] = "voxel"
        elif self.parent.data.fit_data.fit_params.fit_area == "Segmentation":
            self.parent.data.fit_data.fit_segmentation_wise()
            self.parent.data.plt["plt_type"] = "segmentation"


class NNLSFitAction(FitAction):
    def __init__(self, parent: MainWindow):
        """NNLS Fit Action."""
        super().__init__(parent=parent, text="NNLS...", model_name="NNLS")

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(
            self.parent.data.fit_data.fit_params,
            (
                parameters.NNLSbaseParams
                or parameters.NNLSParams
                or parameters.NNLSCVParams
            ),
        ):
            if isinstance(self.parent.data.fit_data.fit_params, parameters.Parameters):
                self.parent.data.fit_data.fit_params = parameters.NNLSParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_NNLS.json",
                    )
                )
            else:
                return

        self.parent.data.fit_data.model_name = "NNLS"

    def check_fit_parameters(self):
        pass


class IVIMFitAction(FitAction):
    def __init__(self, parent: MainWindow, **kwargs):
        """IVIM Fit Action."""
        super().__init__(
            parent=parent,
            text=kwargs.get("text", "IVIM..."),
            model_name=kwargs.get("model_name", "IVIM"),
        )

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(self.parent.data.fit_data.fit_params, parameters.IVIMParams):
            if isinstance(self.parent.data.fit_data.fit_params, parameters.Parameters):
                self.parent.data.fit_data.fit_params = parameters.IVIMParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_IVIM_tri.json",
                    )
                )
            else:
                dialog = FitParametersMessageBox(self.parent.data.fit_data.fit_params)
                if dialog.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.parent.data.fit_data.fit_params = parameters.IVIMParams(
                        Path(
                            self.parent.data.app_path,
                            "resources",
                            "fitting",
                            "default_params_IVIM_tri.json",
                        )
                    )
                else:
                    return None
        self.parent.data.fit_data.model_name = "IVIM"

    def check_fit_parameters(self):
        if self.parent.data.fit_data.fit_params.scale_image == "S/S0":
            self.parent.data.fit_data.fit_params.boundaries[
                "x0"
            ] = self.parent.data.fit_data.fit_params.boundaries["x0"][:-1]
            self.parent.data.fit_data.fit_params.boundaries[
                "lb"
            ] = self.parent.data.fit_data.fit_params.boundaries["lb"][:-1]
            self.parent.data.fit_data.fit_params.boundaries[
                "ub"
            ] = self.parent.data.fit_data.fit_params.boundaries["ub"][:-1]


class IDEALFitAction(IVIMFitAction):
    def __init__(self, parent: MainWindow):
        """IDEAL IVIM Fit Action"""
        super().__init__(parent=parent, text="IDEAL...", model_name="IDEAL")

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(self.parent.data.fit_data.fit_params, parameters.IDEALParams):
            if isinstance(self.parent.data.fit_data.fit_params, parameters.Parameters):
                self.parent.data.fit_data.fit_params = parameters.IDEALParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_IDEAL_tri.json",
                    )
                )
            else:
                dialog = FitParametersMessageBox(self.parent.data.fit_data.fit_params)
                if dialog.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.parent.data.fit_data.fit_params = parameters.IDEALParams(
                        Path(
                            self.parent.data.app_path,
                            "resources",
                            "fitting",
                            "default_params_IDEAL_tri.json",
                        )
                    )
                else:
                    return None
        self.parent.data.fit_data.model_name = "IDEAL"

    def check_fit_parameters(self):
        super().check_fit_parameters()
        if self.parent.data.fit_data.fit_params.scale_image == "S/S0":
            self.parent.data.fit_data.fit_params.tolerance = (
                self.parent.data.fit_data.fit_params.tolerance[:-1]
            )

        if not (
            self.parent.data.fit_data.fit_params.dimension_steps[0]
            == self.parent.data.fit_data.img.array.shape[0:2]
        ).all():
            print(
                f"Matrix size missmatch! {self.parent.data.fit_data.fit_params.dimension_steps[0]} "
                f"vs {self.parent.data.fit_data.img.array.shape[0:2]}"
            )
            dimension_dlg = IDEALDimensionMessageBox()
            if dimension_dlg.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                self.parent.data.fit_data.fit_params.dimension_steps[0] = (
                    self.parent.data.fit_data.img.array.shape[0:2],
                )

    def fit_run(self):
        self.parent.data.fit_data.fit_ideal(
            multi_threading=self.parent.settings.value("multithreading", type=bool)
        )


class SaveResultsToNiftiAction(QAction):
    def __init__(self, parent: MainWindow):
        """Save Results to Nifti file."""
        super().__init__(
            parent=parent,
            text="Save Results to NifTi...",
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.parent = parent
        self.triggered.connect(self.save_results)

    def save_results(self):
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name
        default = file.parent / file.stem / "_" / model / ".nii"
        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Save Results to separate NifTi files",
                default,
            )[0]
        )
        self.parent.data.fit_results.save_fitted_parameters_to_nii(
            file_path,
            shape=self.parent.data.nii_img.array.shape,
            dtype=float,
            parameter_name=self.parent.data.fit_data.fit_params.parameter_name,
        )


class SaveResultsToExcelAction(QAction):
    def __init__(self, parent: MainWindow):
        """Save results to Excel action."""
        super().__init__(
            parent=parent,
            text="Save Results to Excel File...",
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
        slices_contain_seg = self.parent.data.nii_seg.slices_contain_seg

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                "Create and save heatmaps",
                file.parent.__str__() + "\\" + file.stem + "_heatmaps",
            )[0]
        )

        if file_path:
            self.parent.data.fit_data.fit_results.create_heatmap(
                self.parent.data.fit_data, file_path, slices_contain_seg
            )


class FittingMenu(QMenu):
    fit_NNLS: NNLSFitAction
    fit_IVIM: IVIMFitAction
    fit_IDEAL: IDEALFitAction
    save_results_to_excel: SaveResultsToExcelAction
    save_results_to_nifti: SaveResultsToNiftiAction
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
        self.save_results_to_nifti = SaveResultsToNiftiAction(self.parent)
        self.addAction(self.save_results_to_nifti)
        self.save_results_to_excel = SaveResultsToExcelAction(self.parent)
        self.addAction(self.save_results_to_excel)
        self.save_AUC_results = SaveAUCResultsAction(self.parent)
        self.addAction(self.save_AUC_results)
        self.save_spectrum = SaveSpectrumAction(self.parent)
        self.addAction(self.save_spectrum)
        self.create_heat_maps = CreateHeatMapsAction(self.parent)
        self.addAction(self.create_heat_maps)
