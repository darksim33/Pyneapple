from __future__ import annotations

from os import stat
from pathlib import Path

import numpy as np
from numba.core.typeinfer import StaticGetItemConstraint
from scipy import signal

from radimgarray import RadImgArray

from .. import NNLSCVParams, NNLSParams
from ..utils.logger import logger
from .results import BaseResults


class NNLSResults(BaseResults):
    """Class for storing NNLS fitting results."""

    def __init__(self, params: NNLSParams | NNLSCVParams):
        super().__init__(params)
        self.params = params
        self.did_apply_cutoffs = False
        self.cutoffs = []

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
        if self.params.fit_model.reg_order:
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

    def apply_cutoffs(self, cutoffs: list[tuple]) -> None:
        """Apply cutoffs to the results.

        If there are more then one peak in a given interval, geometric mean is used to
        calculate the area under the curve and merge peaks.

        Args:
            cutoffs (list[tuple]): List containing the cutoff values for each range.
        """
        self.cutoffs = cutoffs

        d_dict = self.D
        f_dict = self.f

        for idx in self.D:
            d_values = self.D[idx]
            f_values = self.f[idx]
            new_d = list()
            new_f = list()
            for cutoff in cutoffs:
                indices = np.where(d_values >= cutoff[0]) & (d_values <= cutoff[1])
                _d_values = d_values[indices]
                _f_values = f_values[indices]
                if len(_d_values) == 0:
                    new_d.append(np.nan)
                    new_f.append(np.nan)
                elif len(_d_values) == 1:
                    new_d.append(_d_values[0])
                    new_f.append(_f_values[0])
                else:
                    # if multiple peaks are detected, use geometric mean
                    pos, height = self.geometric_mean_peak(_d_values[0], _f_values[0])
                    new_d.append(pos)
                    new_f.append(height)

            d_dict[idx] = new_d
            f_dict[idx] = new_f
            # curve
            # spectrum

        self.did_apply_cutoffs = True

    @staticmethod
    def geometric_mean_peak(positions, heights):
        """
        Weighted geometric mean when positions are LINEAR values.

        Args:
            positions: array of peak positions in LINEAR space (e.g., D values directly)
            heights: array of peak weights/heights
        Returns:
            mean_position, total_height
        """
        # Weighted geometric mean: GM = exp(Σ(w_i * ln(x_i)) / Σ(w_i))
        # np.exp(np.sum(heights * np.log(positions)) / np.sum(heights))
        weighted_geomean = np.prod(positions ** (heights / np.sum(heights)))

        # Convert back to log space
        mean_position = np.log10(weighted_geomean)
        total_height = np.sum(heights)

        return mean_position, total_height

    def _prepare_non_separate_nii(
        self, file_path: Path, img: RadImgArray, dtype: object | None = None, **kwargs
    ) -> tuple[list[Path], list[RadImgArray]]:
        """Save each parameter in a separate NIfTi file

        Since NNLS can yield varying numbers of peaks, we need to apply cutoffs to the results.

        Args:
            file_path (Path): Path to the NIfTi file.
            img (RadImgArray): The image data.
            dtype (object | None): The data type of the image data.
            kwargs (dict): Additional keyword arguments.
                cutoffs (list[tuple]): List containing the cutoff values for each range.
        """
        if not self.did_apply_cutoffs and not kwargs.get("cutoffs", False):
            logger.warning("No cutoffs applied. Results may be inaccurate or fail.")
        elif not self.did_apply_cutoffs and kwargs.get("cutoffs", False):
            self.apply_cutoffs(kwargs.get("cutoffs", []))
        return super()._prepare_non_separate_nii(file_path, img, dtype, **kwargs)

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
        super().save_spectrum_to_excel(
            file_path,
            bins=bins,
            split_index=split_index,
            is_segmentation=is_segmentation,
            **kwargs,
        )
