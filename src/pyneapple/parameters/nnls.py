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

from ..models import NNLS, NNLSCV
from .parameters import Parameters
from . import NNLSBoundaries
# from nifti import NiiSeg
from radimgarray import RadImgArray, SegImgArray


class NNLSbaseParams(Parameters):
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
        self.fit_function = NNLS.fit
        self.fit_model = NNLS.model

    @property
    def fit_function(self):
        """Returns partial of methods corresponding fit function."""
        return partial(
            self._fit_function,
            basis=self.get_basis(),
            max_iter=self.max_iter,
        )

    @fit_function.setter
    def fit_function(self, method: Callable):
        """Sets fit function."""
        self._fit_function = method

    @property
    def fit_model(self):
        """Returns partial of methods corresponding fit model."""
        return self._fit_model

    @fit_model.setter
    def fit_model(self, method: Callable):
        self._fit_model = method

    def get_basis(self) -> np.ndarray:
        """Calculates the basis matrix for a given set of b-values."""
        basis = np.exp(
            -np.kron(
                self.b_values,
                self.get_bins(),
            )
        )
        return basis

    def eval_fitting_results(self, results: list, **kwargs) -> dict:
        """Determines results for the diffusion parameters d & f out of the
            fitted spectrum.

        Args:
            results (list): List of tuples containing the results of the fitting
                process.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            fitted_results (dict):
                The results of the fitting process combined in a dictionary.
                Each entry holds a dictionary containing the different results.
        """

        spectrum = dict()
        d = dict()
        f = dict()
        curve = dict()

        bins = self.get_bins()
        for element in results:
            spectrum[element[0]] = element[1]

            # Find peaks and calculate fractions
            peak_indexes, properties = signal.find_peaks(element[1], height=0.1)
            f_values = properties["peak_heights"]

            # calculate area under the curve fractions by assuming gaussian curve
            if self.reg_order:
                f_values = self.calculate_area_under_curve(
                    element[1], peak_indexes, f_values
                )

            # Save results and normalise f
            d[element[0]] = bins[peak_indexes]
            f[element[0]] = np.divide(f_values, sum(f_values))

            # Set decay curve
            curve[element[0]] = self.fit_model(
                self.b_values,
                element[1],
                bins,
            )

        fit_results = {
            "d": d,
            "f": f,
            "curve": curve,
            "spectrum": spectrum,
        }

        return fit_results

    @staticmethod
    def calculate_area_under_curve(spectrum: np.ndarray, idx, f_values) -> list:
        """Calculates area under the curve fractions by assuming Gaussian curve.

        Args:
            spectrum (np.ndarray): The spectrum to be analyzed.
            idx (np.ndarray): The indices of the peaks in the spectrum.
            f_values (np.ndarray): The peak heights of the peaks in the
                spectrum.
        Returns:
            f_reg (list): The area under the curve fractions of the peaks in
                the spectrum.
        """
        f_fwhms = signal.peak_widths(spectrum, idx, rel_height=0.5)[0]
        f_reg = list()
        for peak, fwhm in zip(f_values, f_fwhms):
            f_reg.append(
                np.multiply(peak, fwhm)
                / (2 * np.sqrt(2 * np.log(2)))
                * np.sqrt(2 * np.pi)
            )
        return f_reg

    def apply_AUC_to_results(self, fit_results: list) -> tuple[dict, dict]:
        """Takes the fit results and calculates the AUC for each diffusion regime.

        Args:
            fit_results (list): List of tuples containing the results of the fitting
            process.
        Returns:
            d_AUC (dict): The area under the curve of the diffusion coefficients for
                each regime.
            f_AUC (dict): The area under the curve of the fractions for each
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
            raise NotImplementedError(
                "Currently only supports regression orders of 3 or lower"
            )

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

    def get_seg_args(self, img: RadImgArray | np.ndarray, seg: SegImgArray, seg_number: int, *args) -> zip:
        """Adds regularisation and calls parent get_seg_args method.

        Args:
            img (RadImgArray, np.ndarray): Image data.
            seg (SegImgArray): Segmentation data.
            seg_number (int): Segmentation number.
            *args: Additional arguments.
        Returns:
            zip: Zipped list of segmentation arguments.
        """g
        mean_signal = seg.get_mean_signal(img, seg_number)

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
        if hasattr(self, "mu") and getattr(self, "mu") is not None and self.tol is None:
            self.tol = self.mu

        self.fit_function = NNLSCV.fit

    @property
    def fit_function(self):
        """Returns partial of methods corresponding fit function."""
        return partial(
            self._fit_function,
            basis=self.get_basis(),
            max_iter=self.max_iter,
            tol=self.tol,
        )

    @fit_function.setter
    def fit_function(self, method):
        self._fit_function = method

    @property
    def tol(self):
        """Returns the tolerance for the cross validation during fitting."""
        return self._tol

    @tol.setter
    def tol(self, tol):
        self._tol = tol
