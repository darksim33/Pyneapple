from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import signal

from .results import Results, ResultDict
from .. import IVIMParams, IVIMSegmentedParams


class IVIMResults(Results):
    """Class for storing and exoprting IVIM fitting results."""

    def __init__(self, params: IVIMParams):
        super().__init__()
        self.params = params

    def eval_results(self, results: list):
        """Evaluate fitting results.

        Args:
            results (list(tuple(tuple, np.ndarray))): List of fitting results.
        """
        for element in results:
            self.s_0[element[0]] = self._get_s_0(element[1])
            self.f[element[0]] = self._get_fractions(element[1])
            self.d[element[0]] = self._get_diffusion_values(element[1])
            self.t_1[element[0]] = self._get_t_one(element[1])

            self.curve[element[0]] = self.fit_model(
                self.b_values,
                *d[element[0]],
                *f[element[0]],
                S0[element[0]],
                t_one[element[0]],
            )

    def _get_s_0(self, results: np.ndarray) -> np.ndarray:
        """Extract S0 values from the results list."""
        if (
            isinstance(self.params.scale_image, str)
            and self.params.scale_image == "S/S0"
        ):
            return np.ndarray(1)
        else:
            if self.TM:
                return results[-2]
            else:
                return results[-1]

    def _get_fractions(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the fractions of the diffusion components.

        Args:
            results (np.ndarray): Results of the fitting process.
            kwargs (dict):
                n_components (int): set the number of diffusion components manually.
        Returns:
            f_new (np.ndarray): Fractions of the diffusion components.
        """

        n_components = kwargs.get("n_components", self.params.n_components)
        f_new = np.zeros(n_components)
        if (
            isinstance(self.params.scale_image, str)
            and self.params.scale_image == "S/S0"
        ):
            # for S/S0 one parameter less is fitted
            f_new[:n_components] = results[n_components:]
        else:
            if n_components > 1:
                f_new[: n_components - 1] = results[
                    n_components : (2 * n_components - 1)
                ]
            else:
                f_new[0] = 1
        if np.sum(f_new) > 1:  # fit error
            f_new = np.zeros(n_components)
        else:
            f_new[-1] = 1 - np.sum(f_new)
        return f_new

    def _get_diffusion_values(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Extract diffusion values from the results list and add missing.

        Args:
            results (np.ndarray): containing the fitting results
        Returns:
            d_new (np.ndarray): containing all diffusion values
        """
        n_components = kwargs.get("n_components", self.params.n_components)
        return results[:n_components]

    def _get_t_one(self, results: np.ndarray) -> np.ndarray | None:
        """Extract T1 values from the results list."""
        if self.params.TM:
            return results[-1]
        else:
            return None

    @staticmethod
    def _get_bins(number_points: int, limits: tupe[float, float]) -> np.ndarray:
        """Returns range of Diffusion values for NNLS fitting or plotting of Diffusion
        spectra."""
        return np.array(
            np.logspace(
                np.log10(limits[0]),
                np.log10(limits[1]),
                number_points,
            )
        )

    def get_spectrum(
        self,
        d_values: np.ndarray,
        fractions: np.ndarray,
        number_points: int,
        diffusion_range: tuple[float, float],
    ):
        """Calculate the diffusion spectrum for IVIM.

        The diffusion values have to be moved to take diffusion_range and number of
        points into account. The calculated spectrum is store inside the object.

        Args:
            d_values (np.ndarray): Diffusion values of the components.
            fractions (np.ndarray): Fractions of the diffusion components.
            number_points (int): Number of points in the diffusion spectrum.
            diffusion_range (tuple): Range of the diffusion
        """
        bins = self._get_bins(number_points, diffusion_range)
        for pixel in d_values:
            spectrum = np.zeros(number_points)
            for d_value, fraction in zip(d_values[pixel], fractions[pixel]):

                # Diffusion values are moved on range to calculate the spectrum
                index = np.unravel_index(
                    np.argmin(abs(bins - d_value), axis=None),
                    bins.shape,
                )[0].astype(int)

                spectrum += fraction * signal.unit_impulse(number_points, index)
            self.spectrum[pixel] = spectrum


class IVIMSegmentedResults(IVIMResults):
    """Class for storing and exporting segmented IVIM fitting results."""

    def __init__(self, params: IVIMSegmentedParams):
        super().__init__(params)
        self.params = params

    def eval_results(self, results: list):
        """Evaluate fitting results from pixel or segmented fitting.

        Args:
            results (list(tuple(tuple, np.ndarray))): List of fitting results.
        """
        for element in results:
            self.s_0[element[0]] = self._get_s_0(element[1])
            self.f[element[0]] = self._get_fractions(element[1])
            self.d[element[0]] = self._get_diffusion_values(element[1])
            self.t_1[element[0]] = self._get_t_one(element[1])

            self.curve[element[0]] = self.fit_model(
                self.b_values,
                *d[element[0]],
                *f[element[0]],
                S0[element[0]],
                t_one[element[0]],
            )
