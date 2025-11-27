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
                raw.update({(i, j, k): _curve})
                curve.update({(i, j, k): _curve})
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


@pytest.fixture
def results_with_t1(results_biexp_pixel):
    """Add T1 values to biexp results."""
    t1 = {}
    for key in results_biexp_pixel["D"].keys():
        t1.update({key: np.random.uniform(800, 1500)})
    results_biexp_pixel["t1"] = t1
    return results_biexp_pixel


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

    def test_init_creates_empty_dicts(self):
        """Test that initialization creates empty ResultDict instances."""
        results = BaseResults(BaseParams())
        assert len(results.spectrum) == 0
        assert len(results.curve) == 0
        assert len(results.raw) == 0
        assert len(results.D) == 0
        assert len(results.f) == 0
        assert len(results.S0) == 0
        assert len(results.t1) == 0

    def test_load_from_dict(self, results_biexp_pixel):
        """Test loading results from a dictionary."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        assert len(results.D) == len(results_biexp_pixel["D"])
        assert len(results.f) == len(results_biexp_pixel["f"])
        assert len(results.S0) == len(results_biexp_pixel["S0"])
        # Verify data integrity
        for key in results_biexp_pixel["D"].keys():
            assert results.D[key] == results_biexp_pixel["D"][key]

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        results = BaseResults(BaseParams())
        # First update
        results.update_results({"D": {1: [1.0, 2.0]}, "f": {1: [0.3, 0.7]}})
        assert len(results.D) == 1
        # Second update with new data
        results.update_results({"D": {2: [1.5, 2.5]}, "f": {2: [0.4, 0.6]}})
        assert len(results.D) == 2
        assert results.D[1] == [1.0, 2.0]
        assert results.D[2] == [1.5, 2.5]


class TestHelperMethods:
    def test_split_or_not_to_split_no_split_no_seg(self):
        """Test key splitting when both flags are False."""
        key = (1, 2, 3)
        result = BaseResults._split_or_not_to_split(
            key, split_index=False, is_segmentation=False
        )
        assert result == [[1, 2, 3]]

    def test_split_or_not_to_split_with_split(self):
        """Test key splitting when split_index is True."""
        key = (1, 2, 3)
        result = BaseResults._split_or_not_to_split(
            key, split_index=True, is_segmentation=False
        )
        assert result == [1, 2, 3]

    def test_split_or_not_to_split_with_segmentation(self):
        """Test key splitting for segmentation data."""
        key = 5
        result = BaseResults._split_or_not_to_split(
            key, split_index=False, is_segmentation=True
        )
        assert result == [5]

    def test_get_column_names_basic(self):
        """Test column name generation without splitting."""
        results = BaseResults(BaseParams())
        columns = results._get_column_names(split_index=False, is_segmentation=False)
        assert columns == ["pixel"]

    def test_get_column_names_split(self):
        """Test column name generation with split index."""
        results = BaseResults(BaseParams())
        columns = results._get_column_names(split_index=True, is_segmentation=False)
        assert columns == ["x", "y", "z"]

    def test_get_column_names_segmentation(self):
        """Test column name generation for segmentation."""
        results = BaseResults(BaseParams())
        columns = results._get_column_names(split_index=False, is_segmentation=True)
        assert columns == ["seg_number"]

    def test_get_column_names_with_list_additional_cols(self):
        """Test column name generation with additional columns as list."""
        results = BaseResults(BaseParams())
        additional = ["col1", "col2", "col3"]
        columns = results._get_column_names(
            split_index=False, is_segmentation=False, additional_cols=additional
        )
        assert columns == ["pixel", "col1", "col2", "col3"]

    def test_get_column_names_with_1d_array_additional_cols(self):
        """Test column name generation with 1D numpy array."""
        results = BaseResults(BaseParams())
        additional = np.array([0, 50, 100, 150, 200])
        columns = results._get_column_names(
            split_index=False, is_segmentation=False, additional_cols=additional
        )
        assert columns == ["pixel", 0, 50, 100, 150, 200]

    def test_get_column_names_with_2d_array_single_column(self):
        """Test column name generation with 2D array (single column)."""
        results = BaseResults(BaseParams())
        additional = np.array([[1], [2], [3]])
        columns = results._get_column_names(
            split_index=False, is_segmentation=False, additional_cols=additional
        )
        assert columns == ["pixel", 1, 2, 3]

    def test_get_column_names_with_invalid_2d_array(self):
        """Test error handling for invalid 2D array shape."""
        results = BaseResults(BaseParams())
        additional = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Additional columns should be"):
            results._get_column_names(
                split_index=False, is_segmentation=False, additional_cols=additional
            )


class TestExportEXCEL:
    def test_save_to_excel_pixel(self, results_biexp_pixel, out_excel):
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

    def test_save_to_excel_segmentation(self, biexp_results_segmentation, out_excel):
        """Test saving results with segmentation data."""
        results = BaseResults(BaseParams())
        results.load_from_dict(biexp_results_segmentation)
        results.save_to_excel(out_excel, split_index=False, is_segmentation=True)
        assert out_excel.is_file()
        df = pd.read_excel(out_excel, index_col=0)
        assert df.columns.tolist() == ["seg_number", "parameter", "value"]

    def test_save_to_excel_verifies_values(self, results_biexp_pixel, out_excel):
        """Test that saved Excel file contains correct values."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_to_excel(out_excel, split_index=True, is_segmentation=False)
        df = pd.read_excel(out_excel, index_col=0)

        # Check that all D, f, and S0 values are present
        params = df["parameter"].unique()
        assert "S0" in params
        # Count D and f parameters
        d_params = [p for p in params if p.startswith("D_")]
        f_params = [p for p in params if p.startswith("f_")]
        assert len(d_params) == 2
        assert len(f_params) == 2

    def test_save_spectrum_to_excel(self, results_biexp_pixel, out_excel):
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_spectrum_to_excel(out_excel)
        df = pd.read_excel(out_excel, index_col=0)
        for idx, key in enumerate(results.spectrum.keys()):
            spectrum = np.array(df.iloc[idx, 1:])
            assert np.allclose(spectrum, np.squeeze(results.spectrum[key]))

    def test_save_spectrum_to_excel_with_custom_bins(
        self, results_biexp_pixel, out_excel
    ):
        """Test saving spectrum with custom bins."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        # Get spectrum length from first entry
        first_key = list(results.spectrum.keys())[0]
        spectrum_len = len(results.spectrum[first_key])
        custom_bins = np.logspace(-3, -1, spectrum_len)

        results.save_spectrum_to_excel(out_excel, bins=custom_bins)
        df = pd.read_excel(out_excel, index_col=0)
        assert out_excel.is_file()
        # Check that custom bins are used as column names (after the first column)
        for i, bin_val in enumerate(custom_bins):
            assert np.isclose(float(df.columns[i + 1]), bin_val)

    def test_save_spectrum_to_excel_segmentation(
        self, biexp_results_segmentation, out_excel
    ):
        """Test saving spectrum for segmentation data."""
        results = BaseResults(BaseParams())
        results.load_from_dict(biexp_results_segmentation)
        results.save_spectrum_to_excel(
            out_excel, split_index=False, is_segmentation=True
        )
        df = pd.read_excel(out_excel, index_col=0)
        assert df.columns[0] == "seg_number"
        for idx, key in enumerate(results.spectrum.keys()):
            spectrum = np.array(df.iloc[idx, 1:])
            assert np.allclose(spectrum, np.squeeze(results.spectrum[key]))

    def test_save_fit_curve_to_excel(self, results_biexp_pixel, b_values, out_excel):
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_fit_curve_to_excel(out_excel, b_values)
        df = pd.read_excel(out_excel, index_col=0)
        # Verify b_values are in columns
        for b_val in b_values:
            assert b_val in df.columns

        for idx, key in enumerate(results.curve.keys()):
            spectrum = np.array(df.iloc[idx, 1:])
            assert np.allclose(spectrum, np.squeeze(results.curve[key]))

    def test_save_fit_curve_to_excel_split_index(
        self, results_biexp_pixel, b_values, out_excel
    ):
        """Test saving fit curve with split index."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_fit_curve_to_excel(out_excel, b_values, split_index=True)
        df = pd.read_excel(out_excel, index_col=0)
        assert df.columns[0] == "x"
        assert df.columns[1] == "y"
        assert df.columns[2] == "z"

    def test_save_fit_curve_to_excel_segmentation(
        self, biexp_results_segmentation, b_values, out_excel
    ):
        """Test saving fit curve for segmentation."""
        results = BaseResults(BaseParams())
        results.load_from_dict(biexp_results_segmentation)
        results.save_fit_curve_to_excel(out_excel, b_values, is_segmentation=True)
        df = pd.read_excel(out_excel, index_col=0)
        assert df.columns[0] == "seg_number"


# class TestExportNIfTI:
#     """Tests for NIfTI export functionality."""

#     def test_save_to_nii_non_separated(
#         self, results_biexp_pixel, mock_radimgarray, tmp_path
#     ):
#         """Test saving to NIfTI without separate files."""
#         results = BaseResults(BaseParams())
#         results.load_from_dict(results_biexp_pixel)

#         file_path = tmp_path / "test_results"
#         # This will call _save_non_separated_nii
#         # Note: This test assumes mock_radimgarray is properly configured
#         # You may need to adjust based on your actual RadImgArray implementation

#     def test_save_spectrum_to_nii(
#         self, results_biexp_pixel, mock_radimgarray, tmp_path
#     ):
#         """Test saving spectrum to NIfTI."""
#         results = BaseResults(BaseParams())
#         results.load_from_dict(results_biexp_pixel)

#         file_path = tmp_path / "test_spectrum.nii"
#         # Note: This assumes mock_radimgarray is available
#         # You may need to implement this fixture


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

    def test_save_to_hdf5_segmentation(self, biexp_results_segmentation, hdf5_file):
        """Test saving segmentation results to HDF5."""
        results = BaseResults(BaseParams())
        results.load_from_dict(biexp_results_segmentation)
        results.save_to_hdf5(hdf5_file)
        assert hdf5_file.is_file()

    def test_save_to_hdf5_with_t1(self, results_with_t1, hdf5_file):
        """Test saving results with T1 values to HDF5."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_with_t1)
        results.save_to_hdf5(hdf5_file)
        assert hdf5_file.is_file()
        _dict = load_from_hdf5(hdf5_file)
        # Verify T1 values are saved
        assert "t1" in _dict

    def test_save_to_hdf5_excludes_private_attrs(self, results_biexp_pixel, hdf5_file):
        """Test that private attributes are not saved to HDF5."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results._private_attr = "should not be saved"
        results.save_to_hdf5(hdf5_file)
        _dict = load_from_hdf5(hdf5_file)
        assert "_private_attr" not in _dict


class TestEdgeCases:
    def test_empty_results_to_excel(self, out_excel):
        """Test handling of completely empty results."""
        results = BaseResults(BaseParams())
        # Add at least one entry to avoid errors
        results.D.update({(0, 0, 0): []})
        results.f.update({(0, 0, 0): []})
        results.S0.update({(0, 0, 0): 0})
        results.save_to_excel(out_excel)

    def test_single_pixel_results(self, out_excel):
        """Test results with only a single pixel."""
        results = BaseResults(BaseParams())
        results.D.update({(0, 0, 0): [1.0, 2.0]})
        results.f.update({(0, 0, 0): [0.3, 0.7]})
        results.S0.update({(0, 0, 0): 1000})
        results.save_to_excel(out_excel)
        df = pd.read_excel(out_excel, index_col=0)
        # Should have 5 rows: 2 D values + 2 f values + 1 S0
        assert len(df) == 5

    def test_results_with_varying_compartments(self, out_excel):
        """Test results where different pixels have different numbers of compartments."""
        results = BaseResults(BaseParams())
        results.D.update(
            {
                (0, 0, 0): [1.0, 2.0],
                (1, 1, 1): [1.5, 2.5, 3.5],  # 3 compartments
            }
        )
        results.f.update({(0, 0, 0): [0.3, 0.7], (1, 1, 1): [0.2, 0.3, 0.5]})
        results.S0.update({(0, 0, 0): 1000, (1, 1, 1): 1200})
        results.save_to_excel(out_excel)
        assert out_excel.is_file()

    def test_update_with_overlapping_keys(self):
        """Test that updates with overlapping keys overwrite properly."""
        results = BaseResults(BaseParams())
        results.update_results({"D": {1: [1.0, 2.0]}})
        assert results.D[1] == [1.0, 2.0]
        # Update with new value for same key
        results.update_results({"D": {1: [3.0, 4.0]}})
        assert results.D[1] == [3.0, 4.0]

    def test_segmentation_wise_with_empty_identifier(self):
        """Test set_segmentation_wise with empty identifier."""
        results = BaseResults(BaseParams())
        results.D.update({1: [1.0, 2.0]})
        results.set_segmentation_wise({})
        # Should not raise an error

    def test_load_from_dict_partial_data(self):
        """Test loading from dict with only some fields."""
        results = BaseResults(BaseParams())
        partial_data = {
            "D": {(0, 0, 0): [1.0, 2.0]},
            "f": {(0, 0, 0): [0.3, 0.7]},
            # Missing S0, spectrum, curve, etc.
        }
        results.load_from_dict(partial_data)
        assert len(results.D) == 1
        assert len(results.f) == 1
        assert len(results.S0) == 0  # Should remain empty


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_load_update_save_workflow(self, results_biexp_pixel, hdf5_file):
        """Test a complete workflow: load, update, save."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        # Update with additional data
        new_pixel = (99, 99, 99)
        results.update_results(
            {
                "D": {new_pixel: [1.0, 2.0]},
                "f": {new_pixel: [0.4, 0.6]},
                "S0": {new_pixel: 1500},
            }
        )

        # Save and reload
        results.save_to_hdf5(hdf5_file)
        loaded_dict = load_from_hdf5(hdf5_file)

        # Verify the new data is present
        assert new_pixel in loaded_dict["D"]

    def test_segmentation_workflow(
        self, biexp_results_segmentation, out_excel, hdf5_file
    ):
        """Test complete segmentation workflow."""
        results = BaseResults(BaseParams())
        results.load_from_dict(biexp_results_segmentation)

        # Save to both formats
        results.save_to_excel(out_excel, is_segmentation=True)
        results.save_to_hdf5(hdf5_file)

        # Verify both files exist
        assert out_excel.is_file()
        assert hdf5_file.is_file()
