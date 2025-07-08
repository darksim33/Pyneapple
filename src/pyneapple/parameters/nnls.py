"""Parameter classes for the NNLS fitting.
The Parameter classes are the heart of the Pyneapple package. They are used to store
all necessary parameters for the fitting process and to provide the fitting functions
with the necessary arguments. The NNLSbaseParams class is the parent class for both
the NNLSParams and NNLSCVParams classes. It contains the basic methods and attributes
which are further specified in the child classes.

Classes:
    NNLSbaseParams: Basic "private" NNLS Parameter class.
    NNLSParams: NNLS Parameter class for regularized fitting.
    NNLSCVParams: NNLS Parameter class for CV-regularized fitting.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy import signal
from scipy.sparse import diags
from typing import Callable
from functools import partial

from ..utils.logger import logger
from ..models import NNLSModel, NNLSCVModel
from .parameters import BaseParams
from . import NNLSBoundaries

# from nifti import NiiSeg
from radimgarray import RadImgArray, SegImgArray, tools


class NNLSbaseParams(BaseParams):
    """Basic NNLS Parameter class. Parent function for both NNLS and NNLSCV.

    Attributes:
        reg_order (int): Regularisation order for the NNLS fitting.
        boundaries (NNLSBoundaries): Boundaries for the NNLS fitting.
    Methods:
        get_basis()
            Calculates the basis matrix for a given set of b-values.
        eval_fitting_results(results: list, **kwargs)
            Determines results for the diffusion parameters d & f out of the fitted
            spectrum.
        calculate_area_under_curve(spectrum: np.ndarray, idx, f_values)
            Calculates area under the curve fractions by assuming Gaussian curve.
        apply_AUC_to_results(fit_results: list)
            Takes the fit results and calculates the AUC for each diffusion regime.
        get_pixel_args(img: np.ndarray, seg: np.ndarray, *args)
            Applies regularisation to image data and subsequently calls parent
            get_pixel_args method.
        get_seg_args(img: np.ndarray, seg: NiiSeg, seg_number: int, *args)
            Adds regularisation and calls parent get_seg_args
    """

    def __init__(
            self,
            params_json: str | Path | None = None,
    ):
        """Initializes the NNLS parameter class.

        Args:
            params_json (str | Path | None): Path to the json file containing
                the parameters.
        """
        self.reg_order = None
        self.boundaries: NNLSBoundaries = NNLSBoundaries()
        super().__init__(params_json)
        self.fit_model = NNLSModel()

    @property
    def fit_model(self):
        """Returns partial of methods corresponding fit model."""
        return self._fit_model

    @fit_model.setter
    def fit_model(self, method):
        """Sets fitting model."""
        self._fit_model = method

    @property
    def fit_function(self):
        """Returns partial of methods corresponding fit function."""
        return partial(
            self._fit_model.fit,
            basis=self.get_basis(),
            max_iter=self.max_iter,
        )

    def get_bins(self) -> np.ndarray:
        """Returns range of Diffusion values for NNLS fitting or plotting of Diffusion
        spectra."""
        return np.array(
            np.logspace(
                np.log10(self.boundaries.get_axis_limits()[0]),
                np.log10(self.boundaries.get_axis_limits()[1]),
                self.boundaries.number_points,
            )
        )

    def get_basis(self) -> np.ndarray:
        """Calculates the basis matrix for a given set of b-values."""
        basis = np.exp(
            -np.kron(
                self.b_values,
                self.get_bins(),
            )
        )
        return basis

    def apply_AUC_to_results(self, fit_results: list) -> tuple[dict, dict]:
        """Takes the fit results and calculates the AUC for each diffusion regime.

        Args:
            fit_results (list): List of tuples containing the results of the fitting
            process.
        Returns:
            d_AUC (dict): The area under the curve of the diffusion coefficients for
                each regime.
            f_AUC (dict): The area under the curve of the fractions for each

        Note:
            Might not be used!
        """

        regime_boundaries = [0.003, 0.05, 0.3]  # use d_range instead?
        n_regimes = len(regime_boundaries)  # guarantee constant n_entries for heatmaps
        d_AUC, f_AUC = {}, {}

        # Analyse all elements for application of AUC
        for (key, d_values), (_, f_values) in zip(
                fit_results.d.items(), fit_results.f.items()
        ):
            d_AUC[key] = np.zeros(n_regimes)
            f_AUC[key] = np.zeros(n_regimes)

            for regime_idx, regime_boundary in enumerate(regime_boundaries):
                # Check for peaks inside regime
                peaks_in_regime = d_values < regime_boundary

                if not any(peaks_in_regime):
                    continue

                # Merge all peaks within this regime with weighting
                d_regime = d_values[peaks_in_regime]
                f_regime = f_values[peaks_in_regime]
                d_AUC[key][regime_idx] = np.dot(d_regime, f_regime) / sum(f_regime)
                f_AUC[key][regime_idx] = sum(f_regime)

                # Set remaining peaks for analysis of other regimes
                remaining_peaks = d_values >= regime_boundary
                d_values = d_values[remaining_peaks]
                f_values = f_values[remaining_peaks]

        return d_AUC, f_AUC


class NNLSParams(NNLSbaseParams):
    """NNLS Parameter class for regularized fitting.

    Attributes:
        reg_order (int): Regularisation order for the NNLS fitting.
        mu (float): Regularisation parameter for the NNLS fitting.
    Methods:
        get_basis()
            Calculates the basis matrix for a given set of b-values in case of
            regularisation.
        get_pixel_args(img: np.ndarray, seg: np.ndarray, *args)
            Applies regularisation to image data and subsequently calls parent
            get_pixel_args method.
        get_seg_args(img: np.ndarray, seg: NiiSeg, seg_number: int, *args)
            Adds regularisation and calls parent get_seg_args method.
    """

    def __init__(
            self,
            params_json: str | Path | None = None,
    ):
        """Initializes the NNLS parameter class.

        Args:
            params_json (str | Path | None): Path to the json file containing the
                parameters.
        """
        self.reg_order = None
        self.mu = None
        super().__init__(params_json)

    def get_basis(self) -> np.ndarray:
        """Calculates the basis matrix for a given set of b-values in case of
        regularisation."""
        basis = super().get_basis()
        n_bins = self.boundaries.dict["n_bins"]

        if self.reg_order == 0:
            # no reg returns vanilla basis
            reg = np.zeros([n_bins, n_bins])
        elif self.reg_order == 1:
            # weighting with the predecessor
            reg = diags([-1, 1], [0, 1], (n_bins, n_bins)).toarray() * self.mu
        elif self.reg_order == 2:
            # weighting of the nearest neighbours
            reg = diags([1, -2, 1], [-1, 0, 1], (n_bins, n_bins)).toarray() * self.mu
        elif self.reg_order == 3:
            # weighting of the first- and second-nearest neighbours
            reg = (
                    diags([1, 2, -6, 2, 1], [-2, -1, 0, 1, 2], (n_bins, n_bins)).toarray()
                    * self.mu
            )
        else:
            error_msg = f"Currently only supports regression orders of 3 or lower. Got: {self.reg_order}"
            logger.error(error_msg)
            raise NotImplementedError(error_msg)

        # append reg to create regularized NNLS basis
        return np.concatenate((basis, reg))

    def get_pixel_args(self, img: np.ndarray, seg: np.ndarray, *args):
        """Applies regularisation to image data and subsequently calls parent
            get_pixel_args method.

        Args:
            img (np.ndarray): Image data.
            seg (np.ndarray): Segmentation data.
            *args: Additional arguments.
        """
        # Enhance image array for regularisation
        reg = np.zeros(
            (
                np.append(
                    np.array(img.shape[0:3]),
                    self.boundaries.dict.get("n_bins", 0),
                )
            )
        )
        # img_reg = Nii().from_array(np.concatenate((img, reg), axis=3))
        img_reg = np.concatenate((img, reg), axis=3)

        pixel_args = super().get_pixel_args(img_reg, seg)

        return pixel_args

    def get_seg_args(
            self, img: RadImgArray | np.ndarray, seg: SegImgArray, seg_number: int, *args
    ) -> zip:
        """Adds regularisation and calls parent get_seg_args method.

        Args:
            img (RadImgArray, np.ndarray): Image data.
            seg (SegImgArray): Segmentation data.
            seg_number (int): Segmentation number.
            *args: Additional arguments.
        Returns:
            zip: Zipped list of segmentation arguments.
        """
        mean_signal = tools.get_mean_signal(img, seg, seg_number)

        # Enhance image array for regularisation
        reg = np.zeros(self.boundaries.dict.get("n_bins", 0))
        reg_signal = np.concatenate((mean_signal, reg), axis=0)

        return zip([[seg_number]], [reg_signal])


class NNLSCVParams(NNLSbaseParams):
    """NNLS Parameter class for CV-regularized fitting.

    Attributes:
        tol (float): Tolerance for the cross validation during fitting.
    Methods:
        get_basis()
            Calculates the basis matrix for a given set of b-values.
    """

    def __init__(
            self,
            params_json: str | Path | None = None,
    ):
        self.tol = None
        self.reg_order = None
        super().__init__(params_json)
        # if hasattr(self, "mu") and getattr(self, "mu") is not None and self.tol is None:
        #     self.tol = self.mu
        self.fit_model = NNLSCVModel()

    @property
    def fit_function(self):
        """Returns partial of methods corresponding fit function."""
        return partial(
            self._fit_model.fit,
            basis=self.get_basis(),
            max_iter=self.max_iter,
            tol=self.tol,
        )

    @property
    def tol(self):
        """Returns the tolerance for the cross validation during fitting."""
        return self._tol

    @tol.setter
    def tol(self, tol):
        self._tol = tol
