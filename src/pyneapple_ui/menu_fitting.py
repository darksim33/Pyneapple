from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction  # , QIcon

from nifti import Nii, NiiSeg
from .dlg_prompts import (
    FitParametersMessageBox,
    MissingSegmentationMessageBox,
    IDEALSquarePlaneMessageBox,
    IDEALFinalDimensionStepMessageBox,
    RepeatedFitMessageBox,
)
from .dlg_fitting import FittingDlg
from pyneapple import (
    Parameters,
    IVIMParams,
    IVIMSegmentedParams,
    NNLSParams,
    NNLSCVParams,
)

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


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
        self.parent.fit_dlg = FittingDlg(self.parent, self.parent.data.fit_data.params)
        self.parent.fit_dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        run = self.parent.fit_dlg.exec()
        # Load parameters from dialog
        self.parent.data.fit_data.params = (
            self.parent.fit_dlg.parameters.get_parameters()
        )

        # Prepare Data
        # Scale Image if needed
        self.parent.data.fit_data.img = self.parent.data.nii_img.copy()
        self.parent.data.fit_data.img.scale_image(
            self.parent.fit_dlg.params.scale_image
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
            self.update_ui()
            self.parent.data.nii_dyn = Nii().from_array(
                self.parent.data.fit_data.results.spectrum
            )

            # TODO: Change this to spectral dict
            # Save fit results into dynamic nii struct for plotting the spectrum
            self.parent.data.nii_dyn = Nii().from_array(
                self.parent.data.fit_data.results.spectrum
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

    def update_ui(self):
        self.parent.fitting_menu.save_results_to_nifti.setEnabled(True)
        self.parent.fitting_menu.save_results_to_excel.setEnabled(True)
        self.parent.fitting_menu.save_AUC_results.setEnabled(True)
        self.parent.fitting_menu.save_spectrum.setEnabled(True)
        self.parent.fitting_menu.create_heat_maps.setEnabled(True)

    def fit_run(self):
        if self.parent.data.fit_data.params.fit_area == "Pixel":
            self.parent.data.fit_data.fit_pixel_wise(
                multi_threading=self.parent.settings.value("multithreading", type=bool)
            )
            self.parent.data.plt["plt_type"] = "voxel"
        elif self.parent.data.fit_data.params.fit_area == "Segmentation":
            self.parent.data.fit_data.fit_segmentation_wise()
            self.parent.data.plt["plt_type"] = "segmentation"


class NNLSFitAction(FitAction):
    def __init__(self, parent: MainWindow):
        """NNLS Fit Action."""
        super().__init__(parent=parent, text="NNLS...", model_name="NNLS")

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(
            self.parent.data.fit_data.params,
            (NNLSParams or NNLSCVParams),
        ):
            if isinstance(self.parent.data.fit_data.params, Parameters):
                self.parent.data.fit_data.params = NNLSParams(
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

    def update_ui(self):
        # activate UI elements according to fit
        super().update_ui()
        self.parent.fitting_menu.save_results_to_nifti.setEnabled(False)


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
        if not isinstance(self.parent.data.fit_data.params, IVIMParams):
            if isinstance(self.parent.data.fit_data.params, Parameters):
                self.parent.data.fit_data.params = IVIMParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_IVIM_tri.json",
                    )
                )
            else:
                dialog = FitParametersMessageBox(self.parent.data.fit_data.params)
                if dialog.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.parent.data.fit_data.params = IVIMParams(
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
        # S/S0 is now applied while reading the parameters
        # if self.parent.data.fit_data.params.scale_image == "S/S0":
        #     self.parent.data.fit_data.params.boundaries[
        #         "x0"
        #     ] = self.parent.data.fit_data.params.boundaries["x0"][:-1]
        #     self.parent.data.fit_data.params.boundaries[
        #         "lb"
        #     ] = self.parent.data.fit_data.params.boundaries["lb"][:-1]
        #     self.parent.data.fit_data.params.boundaries[
        #         "ub"
        #     ] = self.parent.data.fit_data.params.boundaries["ub"][:-1]
        pass


class IVIMSegmentedFitAction(FitAction):
    def __init__(self, parent: MainWindow, **kwargs):
        """IVIM Fit Action."""
        super().__init__(
            parent=parent,
            text=kwargs.get("text", "IVIM segmented..."),
            model_name=kwargs.get("model_name", "IVIMSegmented"),
        )

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(self.parent.data.fit_data.params, IVIMSegmentedParams):
            if isinstance(self.parent.data.fit_data.params, Parameters):
                self.parent.data.fit_data.params = IVIMSegmentedParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_IVIM_tri.json",
                    )
                )
            else:
                dialog = FitParametersMessageBox(self.parent.data.fit_data.params)
                if dialog.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.parent.data.fit_data.params = IVIMSegmentedParams(
                        Path(
                            self.parent.data.app_path,
                            "resources",
                            "fitting",
                            "default_params_IVIM_tri.json",
                        )
                    )
                else:
                    return None
        self.parent.data.fit_data.model_name = "IVIMSegmented"

    def check_fit_parameters(self):
        pass

    def fit_run(self):
        self.parent.data.fit_data.fit_ivim_segmented(
            multi_threading=self.parent.settings.value("multithreading", type=bool)
        )


class IDEALFitAction(IVIMFitAction):
    def __init__(self, parent: MainWindow):
        """IDEAL IVIM Fit Action"""
        super().__init__(parent=parent, text="IDEAL...", model_name="IDEAL")
        self.setEnabled(False)

    def set_parameter_instance(self):
        """Validate current loaded parameters and change if needed."""
        if not isinstance(self.parent.data.fit_data.params, IDEALParams):
            if isinstance(self.parent.data.fit_data.params, Parameters):
                self.parent.data.fit_data.params = IDEALParams(
                    Path(
                        self.parent.data.app_path,
                        "resources",
                        "fitting",
                        "default_params_IDEAL_tri.json",
                    )
                )
            else:
                dialog = FitParametersMessageBox(self.parent.data.fit_data.params)
                if dialog.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.parent.data.fit_data.params = IDEALParams(
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
        """Check if fit parameters are set properly"""
        super().check_fit_parameters()
        # S0 adjustments
        if self.parent.data.fit_data.params.scale_image == "S/S0":
            self.parent.data.fit_data.params.tolerance = (
                self.parent.data.fit_data.params.tolerance[:-1]
            )
        # Check if image is squared
        if not (
            self.parent.data.fit_data.img.array.shape[0]
            == self.parent.data.fit_data.img.array.shape[1]
        ):
            if (
                IDEALSquarePlaneMessageBox().exec()
                == QtWidgets.QMessageBox.StandardButton.Yes
            ):
                if self.parent.data.nii_seg.path:
                    self.parent.data.fit_data.seg.zero_padding()
                    self.parent.data.fit_data.img.zero_padding()
                    print(
                        f"Padded Image to {self.parent.data.fit_data.img.array.shape[0]}, "
                        f"{self.parent.data.fit_data.img.array.shape[1]}"
                    )
                    # Paste image and segmentation back to main
                    self.parent.data.nii_img.setter(self.parent.data.fit_data.img)
                    self.parent.data.nii_seg.setter(self.parent.data.fit_data.seg)
                    # setup Image
                    self.parent.image_axis.setup_image()

        if not (
            self.parent.data.fit_data.params.dimension_steps[0]
            == self.parent.data.fit_data.img.array.shape[0:2]
        ).all():
            print(
                f"Matrix size missmatch! {self.parent.data.fit_data.params.dimension_steps[0]} "
                f"vs {self.parent.data.fit_data.img.array.shape[0:2]}"
            )
            dimension_dlg = IDEALFinalDimensionStepMessageBox()
            if dimension_dlg.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                self.parent.data.fit_data.params.dimension_steps[0][:] = np.array(
                    self.parent.data.fit_data.img.array.shape[0:2],
                )

    def fit_run(self):
        self.parent.data.fit_data.fit_IDEAL(
            multi_threading=self.parent.settings.value("multithreading", type=bool)
        )


class SaveResultsToNiftiAction(QAction):
    def __init__(self, parent: MainWindow):
        """Save Results to Nifti file."""
        super().__init__(
            parent=parent,
            text="Save Component Results to NifTi...",
            icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
        )
        self.parent = parent
        self.setEnabled(False)
        self.triggered.connect(self.save_results)

    def save_results(self):
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name
        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                caption="Save Results to separate NifTi files",
                directory=(
                    self.parent.data.last_dir / (file.stem + "_" + model + ".nii.gz")
                ).__str__(),
                filter="NifTi (*.nii, *.nii.gz)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent
        self.parent.data.fit_data.results.save_fitted_parameters_to_nii(
            file_path,
            shape=self.parent.data.nii_img.array.shape,
            dtype=float,
            parameter_names=self.parent.data.fit_data.params.boundaries.parameter_names,
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
        self.setEnabled(False)
        self.triggered.connect(self.save)

    def save(self):
        """Saves results to Excel sheet, saved in dir of img file."""
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                caption="Save Results to Excel",
                directory=(
                    self.parent.data.last_dir
                    / (file.stem + "_" + model + "_results.xlsx")
                ).__str__(),
                filter="Excel (*.xlsx)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent

        if file_path:
            if self.parent.data.fit_data.params.fit_area == "Pixel":
                is_segmentation = False
            else:
                is_segmentation = True

            self.parent.data.fit_data.results.save_results_to_excel(
                file_path, is_segmentation=is_segmentation
            )


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
        self.setEnabled(False)
        self.triggered.connect(self.save_AUC)

    def save_AUC(self):
        """Saves results to Excel sheet, saved in dir of img file."""
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                caption="Create and save heatmaps",
                directory=(
                    self.parent.data.last_dir
                    / (file.stem + "_" + model + "_AUC_results.xlsx")
                ).__str__(),
                filter="Excel (*.xlsx)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent

        if file_path:
            (d_AUC, f_AUC,) = self.parent.data.fit_data.params.apply_AUC_to_results(
                self.parent.data.fit_data.results
            )
            self.parent.data.fit_data.results.save_results_to_excel(
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
        self.setEnabled(False)
        self.triggered.connect(self.save_spectrum)

    def save_spectrum(self):
        """Saves spectrum as 4D Nii file."""
        file = self.parent.data.nii_img.path
        model = self.parent.data.fit_data.model_name

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                self.parent,
                caption="Save Spectrum",
                directory=(
                    self.parent.data.last_dir / (file.stem + "_" + model + "_spec.nii")
                ).__str__(),
                filter="NifTi (*.nii, *.nii.gz)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent

        if file_path:
            self.parent.data.fit_data.results.save_spectrum_to_nii(
                file_path, self.parent.data.nii_seg.array.shape
            )


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
                caption="Create and save heatmaps",
                directory=(
                    self.parent.data.last_dir / (file.stem + "_heatmaps.nii")
                ).__str__(),
                filter="NifTi (*.nii, *.nii.gz)",
            )[0]
        )
        self.parent.data.last_dir = Path(file_path).parent

        if file_path:
            self.parent.data.fit_data.results.create_heatmap(
                self.parent.data.fit_data, file_path, slices_contain_seg
            )


class FittingMenu(QMenu):
    fit_NNLS: NNLSFitAction
    fit_IVIM: IVIMFitAction
    fit_IVIM_segmented: IVIMSegmentedFitAction
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
        self.fit_IVIM_segmented = IVIMSegmentedFitAction(self.parent)
        self.addAction(self.fit_IVIM_segmented)
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
