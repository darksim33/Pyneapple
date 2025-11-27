from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from pyneapple.io.hdf5 import load_from_hdf5
from pyneapple.parameters.parameters import BaseParams
from pyneapple.results.results import BaseResults

# from pyneapple.results.types import Results


def get_spectrum(
    D_values: list,
    f_values: list,
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
    bins = np.array(
        np.logspace(
            np.log10(diffusion_range[0]),
            np.log10(diffusion_range[1]),
            number_points,
        )
    )
    spectrum = np.zeros(number_points)
    for d_value, fraction in zip(D_values, f_values):
        # Diffusion values are moved on range to calculate the spectrum
        index = np.unravel_index(
            np.argmin(abs(bins - d_value), axis=None),
            bins.shape,
        )[0].astype(int)

        spectrum += fraction * signal.unit_impulse(number_points, index)
    return spectrum


@pytest.fixture
def results_biexp_pixel(b_values):
    # Set range for biexponential parameters
    f_lower = [10, 10]
    D_lower = [0.0005, 0.005]
    f_upper = [2500, 2500]
    D_upper = [0.003, 0.05]
    # Set shape of biexponential results
    shape = (8, 8, 2)
    n_bins = np.random.randint(250, 500)
    f, D, S0, curve, spectrum, raw = {}, {}, {}, {}, {}, {}
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                f1 = np.random.uniform(f_lower[0], f_upper[0])
                D1 = np.random.uniform(f_lower[1], f_upper[1])
                f2 = np.random.uniform(D_lower[0], D_upper[0])
                D2 = np.random.uniform(D_lower[1], D_upper[1])
                f.update({(i, j, k): [f1, f2]})
                D.update({(i, j, k): [D1, D2]})
                S0.update({(i, j, k): f1 + f2})
                _curve = f1 * np.exp(-D1 * b_values) + f2 * np.exp(-D2 * b_values)
                raw.update({(i, j, k): curve})
                curve.update({(i, j, k): curve})
                _spectrum = get_spectrum(
                    [D1, D2],
                    [f1, f2],
                    n_bins,
                    (min(D_lower), max(D_upper)),
                )
                spectrum.update({(i, j, k): _spectrum})
    return {"f": f, "D": D, "S0": S0, "curve": curve, "raw": raw, "spectrum": spectrum}


@pytest.fixture
def biexp_results_segmentation(b_values) -> dict[str, Any]:
    # Set range for biexponential parameters
    f_lower = [10, 10]
    D_lower = [0.0005, 0.005]
    f_upper = [2500, 2500]
    D_upper = [0.003, 0.05]
    n_segs = np.random.randint(1, 10)
    n_bins = np.random.randint(250, 500)
    f, D, S0, curve, spectrum, raw = {}, {}, {}, {}, {}, {}
    for seg in range(n_segs):
        f1 = np.random.uniform(f_lower[0], f_upper[0])
        D1 = np.random.uniform(f_lower[1], f_upper[1])
        f2 = np.random.uniform(D_lower[0], D_upper[0])
        D2 = np.random.uniform(D_lower[1], D_upper[1])
        f.update({seg: [f1, f2]})
        D.update({seg: [D1, D2]})
        S0.update({seg: f1 + f2})
        _curve = f1 * np.exp(-D1 * b_values) + f2 * np.exp(-D2 * b_values)
        raw.update({seg: _curve})
        curve.update({seg: _curve})
        _spectrum = get_spectrum(
            [D1, D2],
            [f1, f2],
            n_bins,
            (min(D_lower), max(D_upper)),
        )
        spectrum.update({seg: _spectrum})

    return {"f": f, "D": D, "S0": S0, "curve": curve, "raw": raw, "spectrum": spectrum}


class TestBasics:
    def test_custom_dict_validate_key(self):
        results = BaseResults(BaseParams())
        results.spectrum.set_segmentation_wise({(1, 1, 1): 1})
        results.spectrum[np.int32(1)] = 1.1
        for key in results.spectrum:
            assert isinstance(key, int)

    def test_results_update(self):
        f = {1: [1.1, 1.2, 1.3]}
        d = {1: [1.1, 1.2, 1.3]}
        results = BaseResults(BaseParams())
        results.update_results({"D": d, "f": f})
        assert results.D == d
        assert results.f == f

    def test_results_set_seg_wise(self):
        pixel2seg = {(1, 1, 1): 1, (1, 1, 1): 1}
        f = {1: [1.1, 1.2, 1.3]}
        d = {1: [1.1, 1.2, 1.3]}
        results = BaseResults(BaseParams())
        results.f.update(f)
        results.D.update(d)
        results.set_segmentation_wise(pixel2seg)
        assert results.f[1, 1, 1] == f[1]
        assert results.D[1, 1, 1] == d[1]
        assert results.f.identifier == pixel2seg


class TestExportEXCEL:
    def test_save_to_excel(self, results_biexp_pixel, out_excel):
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        # basic
        results.save_to_excel(out_excel, split_index=False, is_segmentation=False)
        assert out_excel.is_file()
        df = pd.read_excel(out_excel, index_col=0)
        assert df.columns.tolist() == [
            "pixel",
            "parameter",
            "value",
        ]
        # Check split index
        out_excel.unlink()
        results.save_to_excel(out_excel, split_index=True, is_segmentation=False)
        df = pd.read_excel(out_excel, index_col=0)
        assert out_excel.is_file()
        assert df.columns.tolist() == [
            "x",
            "y",
            "z",
            "parameter",
            "value",
        ]

    def test_save_spectrum_to_excel(self, results_biexp_pixel, out_excel):
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_spectrum_to_excel(out_excel)
        df = pd.read_excel(out_excel, index_col=0)
        for idx, key in enumerate(results.spectrum.keys()):
            spectrum = np.array(df.iloc[idx, 1:])
            assert np.allclose(spectrum, np.squeeze(results.spectrum[key]))

    def test_save_fit_curve_to_excel(self, results_biexp_pixel, out_excel):
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_spectrum_to_excel(out_excel)
        df = pd.read_excel(out_excel, index_col=0)
        for idx, key in enumerate(results.spectrum.keys()):
            spectrum = np.array(df.iloc[idx, 1:])
            assert np.allclose(spectrum, np.squeeze(results.spectrum[key]))


class TestExportHDF5:
    def compare_dict_to_class(self, _dict, obj):
        for key, value in _dict.items():
            if isinstance(obj, dict):
                class_value = obj[key]
            else:
                class_value = getattr(obj, key)
            if isinstance(class_value, (int, float, str, bool, list, Path)):
                assert value == class_value
            elif isinstance(value, np.ndarray):
                assert np.allclose(value, class_value)
            else:
                self.compare_dict_to_class(value, class_value)

    def test_save_to_hdf5_pixel(self, results_biexp_pixel, hdf5_file):
        results_biexp_pixel.save_to_hdf5(hdf5_file)
        assert hdf5_file.is_file()
        _dict = load_from_hdf5(hdf5_file)
        self.compare_dict_to_class(_dict, results_biexp_pixel)
