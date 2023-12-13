from __future__ import annotations

# from abc import abstractmethod
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction  # , QIcon

from src.utils import Nii, NiiSeg
from src.ui.prompt_dlg import FitParametersDlg, MissingSegDlg
from src.ui.fitting_dlg import FittingDlg, FittingDictionaries
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
        self.triggered.connect(self.fit)

    def b_values_from_dict(self):
        """
        Extract b_values from the fit dialog.

        The b_values_from_dict function is used to extract the b_values from the fit_dict.
        The function first checks if there are any b_values in the fit dict, and if so, it extracts them.
        If they are a string, then it converts them into an array of integers using numpy's fromstring method.
        It then reshapes this array to match that of self.parent.data.fit_data (the data object).
        If they were not a string but instead a list or some other type of iterable object, then we simply convert them into an array using numpy's nparray method.
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

    def fit(self):
        """
        Main pyneapple fitting function for the UI.

        Handles IVIM and NNLS fitting.
        """
        fit_data = self.parent.data.fit_data
        dlg_dict = dict()

        if self.model_name.__contains__("NNLS"):
            if not isinstance(
                fit_data.fit_params,
                (
                    parameters.NNLSParams
                    or parameters.NNLSregParams
                    or parameters.NNLSregCVParams
                ),
            ):
                if isinstance(fit_data.fit_params, parameters.Parameters):
                    fit_data.fit_params = parameters.NNLSregParams()
                else:
                    dialog = FitParametersDlg(fit_data.fit_params)
                    result = dialog.exec()
                    if result:
                        # TODO: Is reg even the right thing to use here @JJ
                        fit_data.fit_params = parameters.NNLSregParams()
                    else:
                        return
            fit_data.model_name = "NNLS"
            dlg_dict = FittingDictionaries.get_nnls_dict(fit_data.fit_params)
        elif self.model_name in ("multiExp", "IVIM"):
            if not isinstance(fit_data.fit_params, parameters.MultiExpParams):
                if isinstance(fit_data.fit_params, parameters.Parameters):
                    fit_data.fit_params = parameters.MultiExpParams()
                else:
                    dialog = FitParametersDlg(fit_data.fit_params)
                    result = dialog.exec()
                    if result:
                        fit_data.fit_params = parameters.MultiExpParams()
                    else:
                        return
            fit_data.model_name = "multiExp"
            dlg_dict = FittingDictionaries.get_multi_exp_dict(fit_data.fit_params)

        # Launch Dlg
        self.parent.fit_dlg = FittingDlg(self.model_name, dlg_dict, fit_data.fit_params)
        self.parent.fit_dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.parent.fit_dlg.exec()

        # Extract Parameters from dlg dict
        b_values = self.b_values_from_dict()
        fit_data.fit_params.b_values = b_values
        self.parent.fit_dlg.dict_to_attributes(fit_data.fit_params)

        fit_data.fit_params.n_pools = self.parent.settings.value(
            "number_of_pools", type=int
        )

        if self.parent.fit_dlg.run:
            if (
                hasattr(fit_data.fit_params, "reg_order")
                and fit_data.fit_params.reg_order == "CV"
            ):
                fit_data.fit_params = parameters.NNLSregCVParams()
                # fit_data.fit_params.model = model.Model.NNLSRegCV()
                self.parent.fit_dlg.dict_to_attributes(fit_data.fit_params)
            elif (
                hasattr(fit_data.fit_params, "reg_order")
                and fit_data.fit_params.reg_order != "CV"
            ):
                fit_data.fit_params.reg_order = int(fit_data.fit_params.reg_order)
                if fit_data.fit_params.reg_order == 0:
                    fit_data.fit_params = parameters.NNLSParams()
                    self.parent.fit_dlg.dict_to_attributes(fit_data.fit_params)
            fit_data.fit_params.b_values = b_values

            self.parent.mainWidget.setCursor(QtCore.Qt.CursorShape.WaitCursor)

            # Prepare Data
            fit_data.img = self.parent.data.nii_img
            fit_data.seg = self.parent.data.nii_seg

            # Check if seg is present else create new one
            if self.parent.data.nii_seg.path:
                fit_data.seg = self.parent.data.nii_seg
            else:
                missing_seg_dlg = MissingSegDlg()
                # result = missing_seg_dlg.exec()
                if missing_seg_dlg.exec():
                    array = np.ones(fit_data.img.array.shape)
                    fit_data.seg = self.parent.data.nii_seg = NiiSeg().from_array(
                        np.expand_dims(array[:, :, :, 1], 3)
                    )

            # Actual Fitting
            if fit_data.fit_params.fit_area == "Pixel":
                fit_data.fit_pixel_wise(
                    multi_threading=self.parent.settings.value(
                        "multithreading", type=bool
                    )
                )
                self.parent.data.nii_dyn = Nii().from_array(
                    self.parent.data.fit_data.fit_results.spectrum
                )

            elif fit_data.fit_area == "Segmentation":
                fit_data.fit_segmentation_wise()

            self.parent.mainWidget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

            self.parent.file_menu.save_fit_image.setEnabled(True)


class NNLSFitAction(FitAction):
    def __init__(self, parent: MainWindow):
        """NNLS Fit Action."""
        super().__init__(parent=parent, text="NNLS...", model_name="NNLS")


class IVIMFitAction(FitAction):
    def __init__(self, parent: MainWindow):
        """IVIM Fit Action."""
        super().__init__(parent=parent, text="IVIM...", model_name="IVIM")


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
            self.parent.data.fit_data.fit_results.save_results(file_path, model)


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
    save_results: SaveResultsAction
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
        """Sets up menu."""
        self.fit_NNLS = NNLSFitAction(self.parent)
        self.addAction(self.fit_NNLS)
        self.fit_IVIM = IVIMFitAction(self.parent)
        self.addAction(self.fit_IVIM)

        self.addSeparator()
        self.save_results = SaveResultsAction(self.parent)
        self.addAction(self.save_results)
        self.create_heat_maps = CreateHeatMapsAction(self.parent)
        self.addAction(self.create_heat_maps)
