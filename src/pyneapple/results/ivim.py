from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

from ..utils.logger import logger
from radimgarray import RadImgArray
from radimgarray.tools import array_to_rgba
from .results import BaseResults
from .. import IVIMParams, IVIMSegmentedParams


class IVIMResults(BaseResults):
    """Class for storing and exporting IVIM fitting results.

    Attributes:
        params (IVIMParams): Parameters for the IVIM fitting.
    """

    def __init__(self, params: IVIMParams):
        super().__init__(params)
        self.params = params

    def eval_results(self, results: list[tuple[tuple, np.ndarray]], **kwargs):
        """Evaluate fitting results.

        Args:
            results (list(tuple(tuple, np.ndarray))): List of fitting results.
        """
        for element in results:
            self.raw[element[0]] = element[1]
            # self.S0[element[0]] = self._get_s0(element[1])
            # self.f[element[0]] = self._get_fractions(element[1]) / self.S0[element[0]]
            self.S0[element[0]], self.f[element[0]] = self._get_contributions(
                element[1]
            )

            self.D[element[0]] = self._get_diffusion_values(element[1])
            self.t1[element[0]] = self._get_t_one(element[1])

            self.curve[element[0]] = self.params.fit_model.model(
                self.params.b_values,
                *self.raw[element[0]],
            )

    def _get_contributions(self, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract S0 and f from results.
        Since they are closely related they are extracted in one step for better
        readability. There are currently 3 cases, depending on the fitting model:
            1. not fit_S0 and not fit_reduced
                Fractions are absolute values and S0 is the sum of all fractions.
            2. fit_S0
                Fractions are relative to S0 and S0 is a free parameter.
            3. fit_reduced
                Fractions are relative to S0 and S0 is fixed to 1 (signal ist normalized).

        Args:
            results (np.ndarray): Fitting results.

        Return:
            tuple[np.ndarray, np.ndarray]: S0 and fractions.
        """

        fit_args = self.params.fit_model.args
        f_positions = [i for i, arg in enumerate(fit_args) if arg.startswith("f")]
        fractions = results[f_positions]

        if self.params.fit_model.fit_reduced or (
            hasattr(self.params.fit_model, "fit_S0") and self.params.fit_model.fit_S0
        ):
            s0 = np.array(1)
            fractions = np.append(fractions, 1 - sum(fractions))
            if (
                hasattr(self.params.fit_model, "fit_S0")
                and self.params.fit_model.fit_S0
            ):
                pos = fit_args.index("S0")
                s0 = results[pos]
        else:
            s0 = np.sum(fractions)
            fractions = fractions / s0
        return (s0, fractions)

    def _get_diffusion_values(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Extract diffusion values from the results list and add missing.

        Args:
            results (np.ndarray): containing the fitting results
        Returns:
            d_new (np.ndarray): containing all diffusion values
        """

        fit_args = self.params.fit_model.args
        d_positions = [i for i, arg in enumerate(fit_args) if arg.startswith("D")]
        return results[d_positions].copy()

    def _get_t_one(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Extract T1 values from the results list."""
        if self.params.fit_model.fit_t1:
            t1_position = self.params.fit_model.args.index("T1")
            return results[t1_position].copy()
        else:
            return np.array([])

    @staticmethod
    def _get_bins(number_points: int, limits: tuple[float, float]) -> np.ndarray:
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
        number_points: int,
        diffusion_range: tuple[float, float],
    ):
        """Calculate the diffusion spectrum for IVIM.

        The diffusion values have to be moved to take diffusion_range and number of
        points into account. The calculated spectrum is store inside the object.

        Args:
            number_points (int): Number of points in the diffusion spectrum.
            diffusion_range (tuple): Range of the diffusion
        """
        bins = self._get_bins(number_points, diffusion_range)
        for pixel in self.D:
            spectrum = np.zeros(number_points)
            for d_value, fraction in zip(self.D[pixel], self.f[pixel]):
                # Diffusion values are moved on range to calculate the spectrum
                index = np.unravel_index(
                    np.argmin(abs(bins - d_value), axis=None),
                    bins.shape,
                )[0].astype(int)

                spectrum += fraction * signal.unit_impulse(number_points, index)
            self.spectrum[pixel] = spectrum

    def _save_separate_nii(
        self, file_path: Path, img: RadImgArray, dtype: object | None = int, **kwargs
    ):
        """Save all fitted parameters to separate NIfTi files.

        Args:
            file_path (Path): Path to the file where the results should be saved.
            img (RadImgArray): Image the fitting was performed on.
            dtype (object, optional): Data type of the saved data. Defaults to int.
            **kwargs: Additional options for saving the data.
                parameter_names (list): List of parameter names to save.
        """

        images = list()
        parameter_names = list()
        d_array = self.D.as_RadImgArray(img)
        f_array = self.f.as_RadImgArray(img)
        for idx in range(self.params.fit_model.n_components):
            images.append(d_array[:, :, :, idx])
            parameter_names.append(f"_d_{idx}")
            images.append(f_array[:, :, :, idx])
            parameter_names.append(f"_f_{idx}")
        if not self.params.fit_model.fit_reduced:
            images.append(self.S0.as_RadImgArray(img))
            parameter_names.append("_s_0")
        if self.params.fit_model.mixing_time:
            images.append(self.t1.as_RadImgArray(img))
            parameter_names.append("_t_1")

        parameter_names = kwargs.get("parameter_names", parameter_names)

        for img, name in zip(images, parameter_names):
            img.save(
                file_path.parent / (file_path.stem + name + ".nii.gz"),
                "nii",
                dtype=dtype,
            )

    def _get_row_data(self, row: list, rows: list, key) -> list:
        rows = super()._get_row_data(row, rows, key)
        if self.params.fit_model.mixing_time:
            rows.append(row + ["T1", self.t1[key]])
        return rows

    def save_heatmap(
        self, file_path: Path, img: RadImgArray, slice_numbers: int | list, **kwargs
    ):
        """Save heatmaps of the diffusion and fraction values.

        Args:
            file_path (Path): Path to save the heatmaps to.
            img (RadImgArray): Image the fitting was performed on.
            slice_numbers (int, list): Slice numbers to save the heatmaps of.
            **kwargs: Additional options for saving the heatmaps.
                alpha (float): Alpha value for the heatmaps.
        """
        if isinstance(slice_numbers, int):
            slice_numbers = [slice_numbers]

        maps = list()
        file_names = list()
        for n_slice in slice_numbers:
            d_map = array_to_rgba(
                self.D.as_RadImgArray(img), alpha=kwargs.get("alpha", 1)
            )
            for idx in range(self.params.fit_model.n_components):
                maps.append(d_map[:, :, :, n_slice, idx])
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_d_{idx}.png")
                )

            f_map = array_to_rgba(
                self.f.as_RadImgArray(img), alpha=kwargs.get("alpha", 1)
            )
            for idx in range(self.params.fit_model.n_components):
                maps.append(f_map[:, :, :, n_slice, idx])
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_f_{idx}.png")
                )
            if not self.params.fit_model.fit_reduced:
                maps.append(
                    array_to_rgba(
                        self.S0.as_RadImgArray(img), alpha=kwargs.get("alpha", 1)
                    )[:, :, :, n_slice]
                )
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_s_0.png")
                )

            if self.params.fit_model.fit_t1:
                t_1_map = array_to_rgba(
                    self.t1.as_RadImgArray(img), alpha=kwargs.get("alpha", 1)
                )[:, :, :, n_slice]
                maps.append(t_1_map)
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_t_1.png")
                )

        for img, name in zip(maps, file_names):
            fig, axs = plt.subplots(1, 1)
            fig.suptitle(f"IVIM {self.params.fit_model.n_components}")
            im = axs.imshow(np.rot90(np.squeeze(img)))
            fig.colorbar(im, ax=axs)
            axs.set_axis_off()
            fig.savefig(name)


class IVIMSegmentedResults(IVIMResults):
    """Class for storing and exporting segmented IVIM fitting results."""

    def __init__(self, params: IVIMSegmentedParams):
        super().__init__(params)
        self.params = params

    def eval_results(self, results: list[tuple[tuple, np.ndarray]], **kwargs):
        """Evaluate fitting results from pixel or segmented fitting.

        Args:
            results (list(tuple(tuple, np.ndarray))): List of fitting results.
            **kwargs: additional necessary options
                fixed_component: list(dict, dict)
                    Dictionary holding results from first fitting step. NOT OPTIONAL

        Returns:
            fitted_results (dict): The results of the fitting process combined in a
                dictionary. Each entry holds a dictionary containing the different
                results.
        """
        try:
            fixed_component = kwargs.get("fixed_component")
        except KeyError:
            error_msg = "No fixed component provided for segmented fitting!"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if fixed_component is None:
            error_msg = "No fixed component provided for segmented fitting!"
            logger.error(error_msg)
            raise ValueError(error_msg)

        for element in results:
            self.S0[element[0]], self.f[element[0]] = self._get_contributions(
                element[1]
            )
            self.D[element[0]] = self._get_diffusion_values(
                element[1], fixed_component=fixed_component[0][element[0]]
            )
            self.t1[element[0]] = self._get_t_one(
                element[1],
                fixed_component=(
                    0 if len(fixed_component) == 1 else fixed_component[1][element[0]]
                ),
            )

            self.curve[element[0]] = self.params.fit_model.model(
                self.params.b_values,
                *self.D[element[0]],
                *self.f[element[0]],
                self.t1[element[0]],
            )

    def _get_contributions(self, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract S0 and f from results.
        Since they are closely related they are extracted in one step for better
        readability. There are currently 3 cases, depending on the fitting model:
            1. not fit_S0 and not fit_reduced
                Fractions are absolute values and S0 is the sum of all fractions.
            2. fit_S0
                Fractions are relative to S0 and S0 is a free parameter.
            3. fit_reduced
                Fractions are relative to S0 and S0 is fixed to 1 (signal ist normalized).

        Args:
            results (np.ndarray): Fitting results.

        Return:
            tuple[np.ndarray, np.ndarray]: S0 and fractions.
        """

        fit_args = self.params.fit_model.args
        f_positions = [i for i, arg in enumerate(fit_args) if arg.startswith("f")]
        fractions = results[f_positions]

        if self.params.fit_model.fit_reduced or (
            hasattr(self.params.fit_model, "fit_S0") and self.params.fit_model.fit_S0
        ):
            s0 = np.array(1)
            fractions = np.append(fractions, 1 - sum(fractions))
            if (
                hasattr(self.params.fit_model, "fit_S0")
                and self.params.fit_model.fit_S0
            ):
                pos = fit_args.index("S0") - 1  # take missing D into account
                s0 = results[pos]
        else:
            s0 = np.sum(fractions)
            fractions = fractions / s0
        return (s0, fractions)

    def _get_diffusion_values(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the diffusion values from the results and adds the fixed component to the results.

        Args:
            results (np.ndarray): containing the fitting results
            **kwargs:
                fixed_component (list | np.ndarray): containing the fixed component
                    results

        Returns:
            d_new (np.ndarray): containing the diffusion values
        """

        fit_args = self.params.fit_model.args
        d_positions = [i for i, arg in enumerate(fit_args) if arg.startswith("D")]

        # Get the position of the fixed component
        fixed_position = (
            self.params.fixed_component
            if self.params.fixed_component is not None
            else -1
        )

        # Filter out the fixed component position from d_positions
        fitted_d_positions = [
            pos for i, pos in enumerate(d_positions) if i != fixed_position
        ]

        # Extract fitted diffusion values
        d_values = results[fitted_d_positions].copy().tolist()

        # Add the fixed component at the correct position
        fixed_result = kwargs.get("fixed_component", 0)
        d_values.insert(fixed_position, fixed_result)
        return np.array(d_values)

    def _get_t_one(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Extract T1 values from the results list.

        Args:
            results (np.ndarray): containing the fitting results
            **kwargs:
                fixed_component (np.ndarray): containing the fixed T1 value on the second array position.
        Returns:
             (np.ndarray): containing the T1 value
        """
        if not self.params.fit_model.fit_t1:
            return np.array([])
        else:
            if self.params.params_1.fit_model.fit_t1:
                try:
                    return kwargs["fixed_component"]
                except KeyError:
                    error_msg = "No fixed T1 component provided for segmented fitting!"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            elif self.params.params_2.fit_model.fit_t1:
                t1_position = self.params.params_2.fit_model.args.index("T1")
                return results[t1_position].copy()
            else:
                error_msg = (
                    "T1 fitting was not configured properly for segmented fitting!"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
