from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

from radimgarray import RadImgArray
from radimgarray.tools import array_to_rgba
from .results import Results
from .. import IVIMParams, IVIMSegmentedParams


class IVIMResults(Results):
    """Class for storing and exporting IVIM fitting results.

    Attributes:
        params (IVIMParams): Parameters for the IVIM fitting.
    """

    def __init__(self, params: IVIMParams):
        super().__init__(params)
        self.params = params

    def eval_results(self, results: list, **kwargs):
        """Evaluate fitting results.

        Args:
            results (list(tuple(tuple, np.ndarray))): List of fitting results.
        """
        for element in results:
            self.s_0[element[0]] = self._get_s_0(element[1])
            self.f[element[0]] = self._get_fractions(element[1])
            self.d[element[0]] = self._get_diffusion_values(element[1])
            self.t_1[element[0]] = self._get_t_one(element[1])

            self.curve[element[0]] = self.params.fit_model(
                self.params.b_values,
                *self.d[element[0]],
                *self.f[element[0]],
                self.s_0[element[0]],
                self.t_1[element[0]],
            )

    def _get_s_0(self, results: np.ndarray) -> np.ndarray:
        """Extract S0 values from the results list."""
        if (
            isinstance(self.params.scale_image, str)
            and self.params.scale_image == "S/S0"
        ):
            return np.ndarray(1)
        else:
            if self.params.TM:
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

    def _get_t_one(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Extract T1 values from the results list."""
        if self.params.TM:
            return results[-1]
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
        for pixel in self.d:
            spectrum = np.zeros(number_points)
            for d_value, fraction in zip(self.d[pixel], self.f[pixel]):
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
        d_array = self.d.as_RadImgArray(img)
        f_array = self.f.as_RadImgArray(img)
        for idx in range(self.params.n_components):
            images.append(d_array[:, :, :, idx])
            parameter_names.append(f"_d_{idx}")
            images.append(f_array[:, :, :, idx])
            parameter_names.append(f"_f_{idx}")
        if not self.params.scale_image == "S/S0":
            images.append(self.s_0.as_RadImgArray(img))
            parameter_names.append("_s_0")
        if self.params.TM:
            images.append(self.t_1.as_RadImgArray(img))
            parameter_names.append("_t_1")

        parameter_names = kwargs.get("parameter_names", parameter_names)

        for img, name in zip(images, parameter_names):
            img.save(
                file_path.parent / (file_path.stem + name + ".nii.gz"),
                "nii",
                dtype=dtype,
            )

    def save_heatmap(
        self, file_path: Path, img: RadImgArray, slice_numbers: int | list
    ):
        """Save heatmaps of the diffusion and fraction values.

        Args:
            file_path (Path): Path to save the heatmaps to.
            img (RadImgArray): Image the fitting was performed on.
            slice_numbers (int, list): Slice numbers to save the heatmaps of.
        """
        if isinstance(slice_numbers, int):
            slice_numbers = [slice_numbers]

        maps = list()
        file_names = list()
        for n_slice in slice_numbers:
            d_map = array_to_rgba(self.d.as_RadImgArray(img))
            for idx in range(self.params.n_components):
                maps.append(d_map[:, :, :, n_slice, idx])
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_d_{idx}.png")
                )

            f_map = array_to_rgba(self.f.as_RadImgArray(img))
            for idx in range(self.params.n_components):
                maps.append(f_map[:, :, :, n_slice, idx])
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_f_{idx}.png")
                )
            if not self.params.scale_image == "S/S0":
                maps.append(
                    array_to_rgba(self.s_0.as_RadImgArray(img))[:, :, :, n_slice]
                )
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_s_0.png")
                )

            if self.params.TM:
                t_1_map = array_to_rgba(self.t_1.as_RadImgArray(img))[:, :, :, n_slice]
                maps.append(t_1_map)
                file_names.append(
                    file_path.parent / (file_path.stem + f"_{n_slice}_t_1.png")
                )

        for img, name in zip(maps, file_names):
            fig, axs = plt.subplots(1, 1)
            fig.suptitle(f"IVIM {self.params.n_components}")
            im = axs.imshow(np.rot90(np.squeeze(img)))
            fig.colorbar(im, ax=axs)
            axs.set_axis_off()
            fig.savefig(name)


class IVIMSegmentedResults(IVIMResults):
    """Class for storing and exporting segmented IVIM fitting results."""

    def __init__(self, params: IVIMSegmentedParams):
        super().__init__(params)
        self.params = params

    def eval_results(self, results: list, **kwargs):
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
            raise ValueError("No fixed component provided for segmented fitting!")
        for element in results:
            self.s_0[element[0]] = self._get_s_0(element[1])
            self.f[element[0]] = self._get_fractions(element[1])
            self.d[element[0]] = self._get_diffusion_values(
                element[1], fixed_component=fixed_component[0][element[0]]
            )
            self.t_1[element[0]] = self._get_t_one(
                element[1],
                fixed_component=0
                if len(fixed_component) == 1
                else fixed_component[1][element[0]],
            )

            self.curve[element[0]] = self.params.fit_model(
                self.params.b_values,
                *self.d[element[0]],
                *self.f[element[0]],
                self.s_0[element[0]],
                self.t_1[element[0]],
            )

    def _get_fractions(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the fractions of the diffusion components for segmented fitting results.

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
                    n_components - 1 : (2 * n_components - 2)
                ]
            else:
                f_new[0] = 1
        if np.sum(f_new) > 1:  # fit error
            f_new = np.zeros(n_components)
        else:
            f_new[-1] = 1 - np.sum(f_new)
        return f_new

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
        fixed_component = kwargs.get("fixed_component", 0)
        d_new = np.zeros(self.params.n_components)
        # assuming that the first component is the one fixed
        d_new[0] = fixed_component
        # since D_slow aka ADC is the default fitting parameter it is always at 0
        # this will cause issues if the fixed component is not the first component
        d_new[1:] = results[: self.params.n_components - 1]
        return d_new

    def _get_t_one(self, results: np.ndarray, **kwargs) -> np.ndarray:
        """Extract T1 values from the results list.

        Args:
            results (np.ndarray): containing the fitting results
            **kwargs:
                fixed_component (np.ndarray): containing the fixed T1 value on the second array position.
        Returns:
             (np.ndarray): containing the T1 value
        """
        fixed = kwargs.get("fixed_component", np.int8(0))
        if not fixed:
            return super()._get_t_one(results)
        else:
            return fixed
