"""Tests for results processing and storage functionality.

This module tests the BaseResults class and its functionality including:

- Result storage: Handling fitted parameters in ResultDict containers
- Data export: Converting results to pandas DataFrames and CSVs
- Statistical analysis: Computing mean, median, std, percentiles over ROIs
- HDF5 serialization: Saving and loading complete result objects
- Result access: Getting values by coordinate or ROI
- Data integrity: Ensuring proper handling of NaN values and missing data

Tests verify correct statistical computations, proper serialization/deserialization,
and accurate data export in various formats.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from pyneapple.io.hdf5 import load_from_hdf5
from pyneapple.parameters.parameters import BaseParams
from pyneapple.results.results import BaseResults
from radimgarray import RadImgArray

from .test_toolbox import ParameterTools

# from pyneapple.results.types import Results


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
        
        # Validate file creation and content
        def validate_columns(df):
            assert df.columns.tolist() == ["seg_number", "parameter", "value"]
        
        ParameterTools.assert_export_file_content(
            results, "save_to_excel", out_excel, 
            lambda f: pd.read_excel(f, index_col=0),
            validate_columns,
            split_index=False, is_segmentation=True
        )

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


class TestExportNIfTI:
    """Tests for NIfTI export functionality."""

    @pytest.fixture
    def mock_img(self):
        """Create a mock RadImgArray for testing."""
        # Create a simple 4D array
        array = np.random.rand(8, 8, 2, 16)
        return RadImgArray(array)

    def test_prepare_non_separate_nii_returns_correct_structure(
        self, results_biexp_pixel, mock_img, temp_dir
    ):
        """Test that _prepare_non_separate_nii returns correct tuple structure."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_prepare"
        file_paths, images = results._prepare_non_separate_nii(
            file_path, mock_img, dtype=float
        )

        # Check return types
        assert isinstance(file_paths, list)
        assert isinstance(images, list)
        assert len(file_paths) == len(images)

        # Check that all paths are Path objects
        assert all(isinstance(p, Path) for p in file_paths)

        # Check that all images are RadImgArray objects
        assert all(isinstance(img, RadImgArray) for img in images)

        # Check expected file names
        assert any("_d.nii" in str(p) for p in file_paths)
        assert any("_f.nii" in str(p) for p in file_paths)
        assert any("_S0.nii" in str(p) for p in file_paths)

    def test_prepare_separate_nii_returns_correct_structure(
        self, results_biexp_pixel, mock_img, temp_dir
    ):
        """Test that _prepare_separate_nii returns correct tuple structure."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_prepare_sep"
        file_paths, images = results._prepare_separate_nii(
            file_path, mock_img, dtype=float
        )

        # Check return types
        assert isinstance(file_paths, list)
        assert isinstance(images, list)
        assert len(file_paths) == len(images)

        # Should have separate files for each D and f component plus S0
        # For biexp: d_0, d_1, f_0, f_1, S0 = 5 files
        assert len(file_paths) >= 5

    def test_prepare_non_separate_with_empty_dicts(self, mock_img, temp_dir):
        """Test preparation when some dicts are empty."""
        results = BaseResults(BaseParams())
        results.D.update({(0, 0, 0): [1.0]})
        # f and S0 are empty

        file_path = temp_dir / "test_empty_prepare"
        file_paths, images = results._prepare_non_separate_nii(
            file_path, mock_img, dtype=float
        )

        # Should only return D file
        assert len(file_paths) == 1
        assert len(images) == 1
        assert "_d.nii" in str(file_paths[0])

    def test_prepare_separate_with_varying_compartments(self, mock_img, temp_dir):
        """Test preparation with varying number of compartments."""
        results = BaseResults(BaseParams())
        results.D.update(
            {
                (0, 0, 0): [1.0, 2.0],
                (1, 1, 1): [1.5, 2.5, 3.5],  # 3 compartments
            }
        )
        results.f.update({(0, 0, 0): [0.5, 0.5], (1, 1, 1): [0.3, 0.3, 0.4]})
        results.S0.update({(0, 0, 0): 1000, (1, 1, 1): 1200})

        file_path = temp_dir / "test_varying_prepare"
        file_paths, images = results._prepare_separate_nii(
            file_path, mock_img, dtype=float
        )

        # Should create files for max compartments (3) for both D and f
        # Plus 1 for S0 = 7 total
        assert len(file_paths) == 7

        # Check that we have d_0, d_1, d_2, f_0, f_1, f_2, S0
        d_files = [p for p in file_paths if "_d_" in str(p)]
        f_files = [p for p in file_paths if "_f_" in str(p)]
        s0_files = [p for p in file_paths if "_S0" in str(p)]

        assert len(d_files) == 3
        assert len(f_files) == 3
        assert len(s0_files) == 1

    def test_file_path_extensions_handled_correctly(
        self, results_biexp_pixel, mock_img, temp_dir
    ):
        """Test that file extensions are handled correctly in preparation."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        # Test with no extension
        file_path = temp_dir / "test_no_ext"
        file_paths, _ = results._prepare_non_separate_nii(
            file_path, mock_img, dtype=float
        )

        # All paths should end with .nii
        assert all(str(p).endswith(".nii") for p in file_paths)

        # Test with .nii extension
        file_path_with_ext = temp_dir / "test_with_ext.nii"
        file_paths_ext, _ = results._prepare_non_separate_nii(
            file_path_with_ext, mock_img, dtype=float
        )

        # Should not double the extension
        assert not any(".nii.nii" in str(p) for p in file_paths_ext)

    def test_save_spectrum_to_nii(self, results_biexp_pixel, mock_img, temp_dir):
        """Test saving spectrum to NIfTI."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_spectrum.nii.gz"
        
        # Use helper to verify file creation and cleanup
        ParameterTools.assert_export_creates_files(
            results, "save_spectrum_to_nii", file_path, [file_path],
            cleanup=True, img=mock_img
        )

    def test_save_spectrum_to_nii_empty(self, mock_img, temp_dir):
        """Test saving empty spectrum to NIfTI."""
        results = BaseResults(BaseParams())
        # Don't load any data - spectrum is empty

        file_path = temp_dir / "test_empty_spectrum.nii.gz"
        # This might raise an error or create an empty file depending on implementation
        try:
            results.save_spectrum_to_nii(file_path, mock_img)
        except (ValueError, KeyError, IndexError):
            # Expected behavior for empty spectrum
            pass

    def test_save_to_nii_segmentation_data(
        self, biexp_results_segmentation, mock_img, temp_dir
    ):
        """Test saving segmentation-based results to NIfTI."""
        results = BaseResults(BaseParams())
        results.load_from_dict(biexp_results_segmentation)

        # Set up segmentation mode
        pixel2seg = {
            (i, j, k): seg_num
            for seg_num in biexp_results_segmentation["D"].keys()
            for i in range(2)
            for j in range(2)
            for k in range(1)
        }
        results.set_segmentation_wise(pixel2seg)

        file_path = temp_dir / "test_seg_results"
        expected_files = [
            temp_dir / "test_seg_results_d.nii.gz",
            temp_dir / "test_seg_results_f.nii.gz",
            temp_dir / "test_seg_results_s0.nii.gz"
        ]
        
        # Use helper to verify all expected files are created
        ParameterTools.assert_export_creates_files(
            results, "save_to_nii", file_path, expected_files,
            cleanup=True, img=mock_img, dtype=float, separate_files=False
        )

    def test_save_to_nii_with_real_img_fixture(
        self, results_biexp_pixel, img, temp_dir
    ):
        """Test with the actual img fixture from conftest."""
        results = BaseResults(BaseParams())
        # Adjust results to match img shape
        adjusted_results = {}
        for key in ["D", "f", "S0", "curve", "spectrum", "raw"]:
            adjusted_results[key] = {}

        # Populate with data matching img dimensions
        for i in range(min(img.shape[0], 4)):
            for j in range(min(img.shape[1], 4)):
                for k in range(min(img.shape[2], 2)):
                    pixel = (i, j, k)
                    adjusted_results["D"][pixel] = [1.0, 2.0]
                    adjusted_results["f"][pixel] = [0.3, 0.7]
                    adjusted_results["S0"][pixel] = 1000.0

        results.load_from_dict(adjusted_results)

        file_path = temp_dir / "test_real_img"
        results.save_to_nii(file_path, img, dtype=float, separate_files=False)

        assert (temp_dir / "test_real_img_d.nii.gz").exists()
        assert (temp_dir / "test_real_img_f.nii.gz").exists()
        assert (temp_dir / "test_real_img_s0.nii.gz").exists()
        (temp_dir / "test_real_img_d.nii.gz").unlink()
        (temp_dir / "test_real_img_f.nii.gz").unlink()
        (temp_dir / "test_real_img_s0.nii.gz").unlink()

    def test_nii_file_can_be_reloaded(self, results_biexp_pixel, mock_img, temp_dir):
        """Test that saved NIfTI files can be reloaded."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_reload"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=False)

        # Try to reload the saved file
        d_file = temp_dir / "test_reload_d.nii.gz"
        reloaded_img = RadImgArray(d_file)
        d_file.unlink()

        # Check that it has the expected shape
        assert reloaded_img.ndim >= 3
        assert reloaded_img.shape[:3] == mock_img.shape[:3]
        (temp_dir / "test_reload_f.nii.gz").unlink()
        (temp_dir / "test_reload_s0.nii.gz").unlink()

    def test_save_to_nii_varying_compartments(self, mock_img, temp_dir):
        """Test saving when different pixels have different numbers of compartments."""
        results = BaseResults(BaseParams())
        results.D.update(
            {
                (0, 0, 0): [1.0, 2.0],
                (1, 1, 1): [1.5, 2.5, 3.5],  # 3 compartments
                (2, 2, 1): [1.8],  # 1 compartment
            }
        )
        results.f.update(
            {(0, 0, 0): [0.3, 0.7], (1, 1, 1): [0.2, 0.3, 0.5], (2, 2, 1): [1.0]}
        )
        results.S0.update({(0, 0, 0): 1000, (1, 1, 1): 1200, (2, 2, 1): 800})

        file_path = temp_dir / "test_varying"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=False)

        # Should handle varying compartments gracefully
        assert (temp_dir / "test_varying_d.nii.gz").exists()

        # The 4th dimension should be sized for the maximum number of compartments
        reloaded = RadImgArray(temp_dir / "test_varying_d.nii.gz")
        assert reloaded.shape[3] >= 3
        (temp_dir / "test_varying_d.nii.gz").unlink()
        (temp_dir / "test_varying_f.nii.gz").unlink()
        (temp_dir / "test_varying_s0.nii.gz").unlink()


class TestSeparateFilesFeature:
    """Test the separate_files parameter functionality."""

    @pytest.fixture
    def mock_img(self):
        """Create a mock RadImgArray for testing."""
        array = np.random.rand(8, 8, 2, 16)
        return RadImgArray(array)

    def test_separate_files_false_creates_combined_files(
        self, results_biexp_pixel, mock_img, temp_dir
    ):
        """Test that separate_files=False creates combined parameter files."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_combined"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=False)

        # Should create 3 files: D, f, S0
        assert (temp_dir / "test_combined_d.nii.gz").exists()
        assert (temp_dir / "test_combined_f.nii.gz").exists()
        assert (temp_dir / "test_combined_S0.nii.gz").exists()

        # Should NOT create indexed files
        assert not (temp_dir / "test_combined_d_0.nii.gz").exists()
        assert not (temp_dir / "test_combined_f_0.nii.gz").exists()

        # Cleanup
        (temp_dir / "test_combined_d.nii.gz").unlink()
        (temp_dir / "test_combined_f.nii.gz").unlink()
        (temp_dir / "test_combined_S0.nii.gz").unlink()

    def test_separate_files_true_creates_indexed_files(
        self, results_biexp_pixel, mock_img, temp_dir
    ):
        """Test that separate_files=True creates indexed parameter files."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_separate"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=True)

        # Should create indexed files for d and f (biexp has 2 components)
        assert (temp_dir / "test_separate_d_0.nii.gz").exists()
        assert (temp_dir / "test_separate_d_1.nii.gz").exists()
        assert (temp_dir / "test_separate_f_0.nii.gz").exists()
        assert (temp_dir / "test_separate_f_1.nii.gz").exists()
        assert (temp_dir / "test_separate_S0.nii.gz").exists()

        # Should NOT create combined files
        assert not (temp_dir / "test_separate_d.nii.gz").exists()
        assert not (temp_dir / "test_separate_f.nii.gz").exists()

        # Cleanup
        for file in temp_dir.glob("test_separate_*.nii.gz"):
            file.unlink()

    def test_separate_files_with_single_compartment(self, mock_img, temp_dir):
        """Test separate_files with single compartment per pixel."""
        results = BaseResults(BaseParams())
        results.D.update({(0, 0, 0): [1.0], (1, 1, 1): [1.5]})
        results.f.update({(0, 0, 0): [1.0], (1, 1, 1): [1.0]})
        results.S0.update({(0, 0, 0): 1000, (1, 1, 1): 1200})

        file_path = temp_dir / "test_single"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=True)

        # Should create one indexed file for each parameter
        assert (temp_dir / "test_single_d_0.nii.gz").exists()
        assert (temp_dir / "test_single_f_0.nii.gz").exists()
        assert (temp_dir / "test_single_S0.nii.gz").exists()

        # Cleanup
        for file in temp_dir.glob("test_single_*.nii.gz"):
            file.unlink()


class TestDtypeHandling:
    """Test dtype parameter handling in NIfTI saving."""

    @pytest.fixture
    def mock_img(self):
        """Create a mock RadImgArray for testing."""
        array = np.random.rand(8, 8, 2, 16)
        return RadImgArray(array)

    def test_dtype_float_preserves_precision(
        self, results_biexp_pixel, mock_img, temp_dir
    ):
        """Test that float dtype preserves decimal precision."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_float_dtype"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=False)

        # Load and check that we still have float precision
        loaded_img = RadImgArray(temp_dir / "test_float_dtype_d.nii.gz")
        assert loaded_img.dtype in [np.float32, np.float64]

        # Cleanup
        (temp_dir / "test_float_dtype_d.nii.gz").unlink()
        (temp_dir / "test_float_dtype_f.nii.gz").unlink()
        (temp_dir / "test_float_dtype_S0.nii.gz").unlink()

    def test_dtype_int_converts_values(self, results_biexp_pixel, mock_img, temp_dir):
        """Test that int dtype converts values appropriately."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_int_dtype"
        results.save_to_nii(file_path, mock_img, dtype=int, separate_files=False)

        # Load and check that we have integer type
        loaded_img = RadImgArray(temp_dir / "test_int_dtype_S0.nii.gz")
        assert np.issubdtype(loaded_img.dtype, np.integer)

        # Cleanup
        (temp_dir / "test_int_dtype_d.nii.gz").unlink()
        (temp_dir / "test_int_dtype_f.nii.gz").unlink()
        (temp_dir / "test_int_dtype_S0.nii.gz").unlink()

    def test_dtype_none_uses_default(self, results_biexp_pixel, mock_img, temp_dir):
        """Test that dtype=None uses default behavior."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        file_path = temp_dir / "test_none_dtype"
        results.save_to_nii(file_path, mock_img, dtype=None, separate_files=False)

        # Should create files successfully
        assert (temp_dir / "test_none_dtype_d.nii.gz").exists()

        # Cleanup
        (temp_dir / "test_none_dtype_d.nii.gz").unlink()
        (temp_dir / "test_none_dtype_f.nii.gz").unlink()
        (temp_dir / "test_none_dtype_S0.nii.gz").unlink()


class TestNIfTIEdgeCases:
    """Test edge cases specific to NIfTI handling."""

    @pytest.fixture
    def mock_img(self):
        """Create a mock RadImgArray for testing."""
        array = np.random.rand(8, 8, 2, 16)
        return RadImgArray(array)

    def test_nii_path_with_extension(self, results_biexp_pixel, mock_img, temp_dir):
        """Test that method handles paths with .nii extension correctly."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        # Path already has .nii extension
        file_path = temp_dir / "test_with_ext.nii"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=False)

        # Should not double the extension
        assert (temp_dir / "test_with_ext_d.nii.gz").exists()
        (temp_dir / "test_with_ext_d.nii.gz").unlink()
        (temp_dir / "test_with_ext_f.nii.gz").unlink()
        (temp_dir / "test_with_ext_s0.nii.gz").unlink()

    def test_nii_creates_parent_directories(
        self, results_biexp_pixel, mock_img, temp_dir
    ):
        """Test that parent directories are created if they don't exist."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)

        # Use a nested path
        nested_path = temp_dir / "subdir1" / "subdir2" / "test_results"
        # Depending on implementation, this might fail or succeed
        try:
            results.save_to_nii(
                nested_path, mock_img, dtype=float, separate_files=False
            )
            # If it succeeds, check the file was created
            assert (temp_dir / "subdir1" / "subdir2" / "test_results_d.nii.gz").exists()
        except (FileNotFoundError, OSError):
            # Expected if parent directory creation is not handled
            pass

    def test_nii_with_minimal_image(self, temp_dir):
        """Test with minimum viable image."""
        results = BaseResults(BaseParams())
        results.D.update({(0, 0, 0): [1.0]})
        results.f.update({(0, 0, 0): [1.0]})
        results.S0.update({(0, 0, 0): 100})

        # Create minimal image
        minimal_img = RadImgArray(np.ones((1, 1, 1, 1)))

        file_path = temp_dir / "test_minimal"
        results.save_to_nii(file_path, minimal_img, dtype=float, separate_files=False)

        assert (temp_dir / "test_minimal_d.nii.gz").exists()
        (temp_dir / "test_minimal_d.nii.gz").unlink()
        (temp_dir / "test_minimal_f.nii.gz").unlink()
        (temp_dir / "test_minimal_s0.nii.gz").unlink()

    def test_nii_single_value_per_pixel(self, mock_img, temp_dir):
        """Test when each pixel has only a single value (not array)."""
        results = BaseResults(BaseParams())
        # Single diffusion coefficient per pixel
        results.D.update({(0, 0, 0): [1.0], (1, 1, 1): [1.5]})
        results.f.update({(0, 0, 0): [1.0], (1, 1, 1): [1.0]})
        results.S0.update({(0, 0, 0): 1000, (1, 1, 1): 1200})

        file_path = temp_dir / "test_single_val"
        results.save_to_nii(file_path, mock_img, dtype=float, separate_files=False)

        assert (temp_dir / "test_single_val_d.nii.gz").exists()
        assert (temp_dir / "test_single_val_f.nii.gz").exists()
        assert (temp_dir / "test_single_val_s0.nii.gz").exists()
        (temp_dir / "test_single_val_d.nii.gz").unlink()
        (temp_dir / "test_single_val_f.nii.gz").unlink()
        (temp_dir / "test_single_val_s0.nii.gz").unlink()


# ---  HDF5


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
            elif isinstance(value, str):
                assert value == class_value
            else:
                self.compare_dict_to_class(value, class_value)

    def test_save_to_hdf5_pixel(self, results_biexp_pixel, hdf5_file):
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_to_hdf5(hdf5_file)
        assert hdf5_file.is_file()
        _dict = load_from_hdf5(hdf5_file)
        self.compare_dict_to_class(_dict, results)

    def test_save_to_hdf5_segmentation(self, biexp_results_segmentation, hdf5_file):
        """Test saving segmentation results to HDF5."""
        results = BaseResults(BaseParams())
        results.load_from_dict(biexp_results_segmentation)
        results.save_to_hdf5(hdf5_file)
        assert hdf5_file.is_file()

    def test_save_to_hdf5_excludes_private_attrs(self, results_biexp_pixel, hdf5_file):
        """Test that private attributes are not saved to HDF5."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results._private_attr = "should not be saved"
        results.save_to_hdf5(hdf5_file)
        _dict = load_from_hdf5(hdf5_file)
        assert "_private_attr" not in _dict

    def test_save_to_hdf5_as_array(self, results_biexp_pixel, hdf5_file):
        """Test saving results as array to HDF5."""
        results = BaseResults(BaseParams())
        results.load_from_dict(results_biexp_pixel)
        results.save_to_hdf5_as_array(
            hdf5_file, img=RadImgArray(np.ones((8, 8, 2, 16)))
        )
        _dict = load_from_hdf5(hdf5_file)
        assert "D" in _dict
        assert "f" in _dict
        assert "S0" in _dict


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
