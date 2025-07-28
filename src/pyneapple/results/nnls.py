from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import signal

from radimgarray import RadImgArray
from .results import BaseResults
from .. import NNLSParams, NNLSCVParams


class NNLSResults(BaseResults):
    """Class for storing NNLS fitting results."""

    def __init__(self, params: NNLSParams | NNLSCVParams):
        super().__init__(params)
        self.params = params

    def eval_results(self, results: list[tuple[tuple, np.ndarray]], **kwargs):
        """Evaluate fitting results.

        Args:
            results (list(tuple(tuple, np.ndarray))): List of fitting results.
        """
        for element in results:
            self.spectrum[element[0]] = element[1]

            self.D[element[0]], self.f[element[0]] = self._get_peak_stats(element[1])
            self.S0[element[0]] = np.array(1)

            self.curve[element[0]] = self.params.fit_model.model(
                b_values=self.params.b_values,
                spectrum=element[1],
                bins=self.params.get_bins(),
            )

    def _get_peak_stats(self, spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get fractions and D values from the spectrum.

        Args:
            spectrum (np.ndarray): Spectrum from fit.
        Returns:
            fractions, d_values (tuple[np.ndarray, np.ndarray]): Fractions and D values.
        """
        # find peaks and calculate fractions
        peak_indexes, properties = signal.find_peaks(spectrum, height=0.1)

        f_values = properties["peak_heights"]
        if self.params.reg_order:
            # Correct fractions for regularized spectra
            f_values = self._calculate_area_under_curve(
                spectrum, peak_indexes, f_values
            )

        # adjust peak fractions
        f_values = np.divide(f_values, np.sum(f_values))

        d_values = self.params.get_bins()[peak_indexes]
        return f_values, d_values

    @staticmethod
    def _calculate_area_under_curve(spectrum: np.ndarray, idx, f_values) -> list:
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

    # def _get_column_names(
    #     self,
    #     split_index: bool = False,
    #     is_segmentation: bool = False,
    #     additional_cols: list = None,
    # ) -> list:
    #     column_names = super()._get_column_names(split_index, is_segmentation)
    #     return column_names + self.params.boundaries.get_boundary_names()

    def _save_separate_nii(
        self, file_path: Path, img: RadImgArray, dtype: object | None = ..., **kwargs
    ):
        # TODO: Implement saving of NNLS results
        return super()._save_separate_nii(file_path, img, dtype, **kwargs)

    def save_spectrum_to_excel(
        self,
        file_path: Path | str,
        bins: list | np.ndarray = list(),
        split_index: bool = False,
        is_segmentation: bool = False,
        **kwargs,
    ):
        """Save the spectrum to an Excel file.

        Args:
            file_path (Path | str): Path to the Excel file.
            split_index (bool): Whether to split the index into separate columns.
            is_segmentation (bool): Whether the data is a segmentation.
            **kwargs: Additional keyword arguments.
        """
        bins = self.params.get_bins() if len(bins) == 0 else bins
        super().save_spectrum_to_excel(file_path, bins=bins, **kwargs)
