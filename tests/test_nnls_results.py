"""Tests for NNLS (Non-Negative Least Squares) results processing.

This module tests the NNLSResults class functionality including:

- Result storage: Storing NNLS fitted spectra and coefficients
- Statistical analysis: Computing statistics over ROIs and spectra
- Data export: Converting NNLS results to DataFrames and CSVs
- Spectrum processing: Handling continuous diffusion spectra
- HDF5 serialization: Saving and loading NNLS result objects
- Integration: Working with NNLSParams and basis matrices

NNLS fitting produces diffusion spectra (distributions) rather than discrete
components, requiring specialized result handling and analysis methods.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyneapple import NNLSParams, NNLSResults
from radimgarray import RadImgArray


def test_nnls_eval_fitting_results(nnls_fit_results, nnls_params, seg_reduced):
    fit_results = NNLSResults(nnls_params)
    fit_results.eval_results(nnls_fit_results[0])
    for idx in nnls_fit_results[3]:
        assert fit_results.f[idx].all() == nnls_fit_results[2][idx].all()
        assert fit_results.D[idx].all() == nnls_fit_results[1][idx].all()


def test_nnls_apply_auc(nnls_params, nnls_fit_results, seg_reduced):
    fit_results = NNLSResults(nnls_params)
    fit_results.eval_results(nnls_fit_results[0])
    assert nnls_params.apply_AUC_to_results(fit_results)


class TestNNLSCutoffHandling:
    """Comprehensive tests for NNLS cutoff functionality."""

    def test_apply_cutoffs_basic(self, nnls_params, nnls_fit_results):
        """Test basic cutoff application."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        cutoffs = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)]
        fit_results.apply_cutoffs(cutoffs)

        assert fit_results.did_apply_cutoffs is True
        assert fit_results.cutoffs == cutoffs

        # After cutoffs, each pixel should have len(cutoffs) compartments
        for pixel in fit_results.D.keys():
            assert len(fit_results.D[pixel]) == len(cutoffs)
            assert len(fit_results.f[pixel]) == len(cutoffs)

    def test_apply_cutoffs_preserves_total_fraction(self, nnls_params):
        """Test that total fraction is preserved after cutoffs."""
        fit_results = NNLSResults(nnls_params)

        # Create controlled spectrum with known peaks in desired ranges
        bins = nnls_params.get_bins()
        n_bins = len(bins)

        # Define cutoffs that we'll use
        cutoffs = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)]

        # Create spectrum with peaks in each cutoff range
        spectrum = np.zeros(n_bins)

        # Peak 1: randomly in range (0.001, 0.01) with fraction 0.3
        idx1 = np.where((bins >= cutoffs[0][0]) & (bins <= cutoffs[0][1]))[0]
        if len(idx1) > 0:
            spectrum[np.random.choice(idx1)] = 0.3

        # Peak 2: randomly in range (0.01, 0.05) with fraction 0.5
        idx2 = np.where((bins >= cutoffs[1][0]) & (bins <= cutoffs[1][1]))[0]
        if len(idx2) > 0:
            spectrum[np.random.choice(idx2)] = 0.5

        # Peak 3: randomly in range (0.05, 0.2) with fraction 0.2
        idx3 = np.where((bins >= cutoffs[2][0]) & (bins <= cutoffs[2][1]))[0]
        if len(idx3) > 0:
            spectrum[np.random.choice(idx3)] = 0.2

        # Normalize spectrum to sum to 1.0
        if spectrum.sum() > 0:
            spectrum = spectrum / spectrum.sum()

        # Evaluate results with our controlled spectrum
        fit_results.eval_results([((0, 0, 0), spectrum)])

        # Store initial total fraction
        initial_f_sum = np.sum(fit_results.f[(0, 0, 0)])

        # Apply cutoffs
        fit_results.apply_cutoffs(cutoffs)

        # Check that total fraction is preserved (accounting for NaN values)
        f_values = np.array(fit_results.f[(0, 0, 0)])
        valid_fractions = f_values[~np.isnan(f_values)]

        # Should have fractions in all three ranges (no NaN)
        assert len(valid_fractions) == len(cutoffs), (
            "Should have peaks in all cutoff ranges"
        )

        # Total fraction should be approximately 1.0
        assert np.isclose(np.sum(valid_fractions), 1.0, atol=0.01), (
            f"Expected sum ≈ 1.0, got {np.sum(valid_fractions)}"
        )

        # Should also be close to the initial sum
        assert np.isclose(np.sum(valid_fractions), initial_f_sum, atol=0.01), (
            f"Fraction sum changed: {initial_f_sum} -> {np.sum(valid_fractions)}"
        )

    def test_apply_cutoffs_handles_no_peaks_in_range(self, nnls_params):
        """Test cutoffs when some ranges have no peaks."""
        fit_results = NNLSResults(nnls_params)
        peak_pos = 175
        # Create a spectrum with only one peak at a specific location
        spectrum = np.zeros(350)
        spectrum[peak_pos] = 1.0  # Peak in middle

        fit_results.eval_results([((0, 0, 0), spectrum)])

        # Define cutoffs where some don't contain the peak
        bins = nnls_params.get_bins()
        mid_value = bins[peak_pos]
        cutoffs = [
            (bins[0], mid_value * 0.5),  # Below peak - should be NaN
            (mid_value * 0.8, mid_value * 1.2),  # Contains peak
            (mid_value * 2, bins[-1]),  # Above peak - should be NaN
        ]

        fit_results.apply_cutoffs(cutoffs)

        # Check that empty ranges have NaN
        assert np.isnan(fit_results.D[(0, 0, 0)][0])
        assert not np.isnan(fit_results.D[(0, 0, 0)][1])
        assert np.isnan(fit_results.D[(0, 0, 0)][2])

    def test_apply_cutoffs_handles_multiple_peaks_in_range(self, nnls_params):
        """Test cutoffs when a range contains multiple peaks."""
        fit_results = NNLSResults(nnls_params)

        # Create a spectrum with two peaks close together
        spectrum = np.zeros(350)
        spectrum[30] = 0.6  # First peak
        spectrum[35] = 0.4  # Second peak

        fit_results.eval_results([((0, 0, 0), spectrum)])

        bins = nnls_params.get_bins()
        # Define a cutoff that contains both peaks
        cutoffs = [(bins[25], bins[40])]

        fit_results.apply_cutoffs(cutoffs)

        # Should merge the peaks using geometric mean
        assert len(fit_results.D[(0, 0, 0)]) == 1
        assert not np.isnan(fit_results.D[(0, 0, 0)][0])
        assert not np.isnan(fit_results.f[(0, 0, 0)][0])

    def test_apply_cutoffs_overlapping_ranges_not_recommended(
        self, nnls_params, nnls_fit_results
    ):
        """Test behavior with overlapping cutoff ranges (edge case)."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        # Overlapping cutoffs (not recommended but should still work)
        cutoffs = [(0.0, 0.005), (0.003, 0.01), (0.008, 0.1)]
        fit_results.apply_cutoffs(cutoffs)

        # Should still apply cutoffs
        assert fit_results.did_apply_cutoffs is True
        for pixel in fit_results.D.keys():
            assert len(fit_results.D[pixel]) == len(cutoffs)

    def test_cutoffs_attribute_stores_values(self, nnls_params, nnls_fit_results):
        """Test that cutoffs are stored in the results object."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        cutoffs = [(0.0, 0.001), (0.001, 0.01)]
        fit_results.apply_cutoffs(cutoffs)

        assert fit_results.cutoffs == cutoffs
        assert fit_results.did_apply_cutoffs is True


# ========== NIfTI Preparation Tests ==========


class TestNNLSNIfTIPreparation:
    """Test NNLS-specific NIfTI preparation methods."""

    def test_prepare_non_separate_nii_with_cutoffs_applied(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test that _prepare_non_separate_nii works correctly after cutoffs."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        cutoffs = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)]
        fit_results.apply_cutoffs(cutoffs)

        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        file_path = temp_dir / "test_cutoffs"

        file_paths, images = fit_results._prepare_non_separate_nii(
            file_path, mock_img, dtype=float
        )

        assert len(file_paths) > 0
        assert len(images) > 0
        assert fit_results.did_apply_cutoffs is True

        # Check expected files
        assert any("_d.nii" in str(p) for p in file_paths)
        assert any("_f.nii" in str(p) for p in file_paths)
        assert any("_S0.nii" in str(p) for p in file_paths)

    def test_prepare_nii_without_cutoffs_logs_warning(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test that warning is logged when cutoffs are not applied."""
        import sys
        from io import StringIO

        from pyneapple.utils.logger import logger

        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        file_path = temp_dir / "test_no_cutoffs"

        # Capture loguru output
        string_io = StringIO()
        logger_id = logger.add(string_io, level="WARNING", format="{message}")

        try:
            # Prepare without applying cutoffs first
            file_paths, images = fit_results._prepare_non_separate_nii(
                file_path, mock_img, dtype=float
            )

            # Get logged messages
            log_output = string_io.getvalue()

            # Check that warning was logged
            assert "cutoff" in log_output.lower(), (
                f"Expected warning about cutoffs, got: {log_output}"
            )
        finally:
            # Remove the handler
            logger.remove(logger_id)

    def test_prepare_nii_applies_cutoffs_from_kwargs(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test that cutoffs can be passed via kwargs to preparation method."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        cutoffs = [(0.0, 0.001), (0.001, 0.01)]
        file_path = temp_dir / "test_kwargs_cutoffs"

        file_paths, images = fit_results._prepare_non_separate_nii(
            file_path, mock_img, dtype=float, cutoffs=cutoffs
        )

        # Cutoffs should now be applied
        assert fit_results.did_apply_cutoffs is True
        assert fit_results.cutoffs == cutoffs

    def test_prepare_nii_doesnt_reapply_cutoffs_if_already_applied(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test that cutoffs aren't reapplied if already done."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        first_cutoffs = [(0.0, 0.001), (0.001, 0.01)]
        fit_results.apply_cutoffs(first_cutoffs)

        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        file_path = temp_dir / "test_no_reapply"

        # Try to pass different cutoffs via kwargs
        second_cutoffs = [(0.0, 0.005), (0.005, 0.1)]
        file_paths, images = fit_results._prepare_non_separate_nii(
            file_path, mock_img, dtype=float, cutoffs=second_cutoffs
        )

        # Original cutoffs should be preserved
        assert fit_results.cutoffs == first_cutoffs


# ========== Full NIfTI Saving Tests ==========


class TestNNLSNIfTISaving:
    """Test complete NIfTI saving workflow for NNLS results."""

    def test_save_to_nii_with_cutoffs_from_kwargs(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test full save_to_nii workflow with cutoffs passed as kwargs."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        cutoffs = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)]
        file_path = temp_dir / "test_full_cutoffs"

        fit_results.save_to_nii(
            file_path, mock_img, dtype=float, separate_files=False, cutoffs=cutoffs
        )

        # Files should be created
        assert (temp_dir / "test_full_cutoffs_d.nii.gz").exists()
        assert (temp_dir / "test_full_cutoffs_f.nii.gz").exists()
        assert (temp_dir / "test_full_cutoffs_S0.nii.gz").exists()

        # Cleanup
        for file in temp_dir.glob("test_full_cutoffs_*.nii.gz"):
            file.unlink()

    def test_save_to_nii_separate_files_with_cutoffs(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test saving with separate_files=True after applying cutoffs."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        cutoffs = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)]
        fit_results.apply_cutoffs(cutoffs)

        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        file_path = temp_dir / "test_separate_cutoffs"

        fit_results.save_to_nii(file_path, mock_img, dtype=float, separate_files=True)

        # Should create separate files for each cutoff range
        for idx in range(len(cutoffs)):
            assert (temp_dir / f"test_separate_cutoffs_d_{idx}.nii.gz").exists()
            assert (temp_dir / f"test_separate_cutoffs_f_{idx}.nii.gz").exists()

        assert (temp_dir / "test_separate_cutoffs_S0.nii.gz").exists()

        # Cleanup
        for file in temp_dir.glob("test_separate_cutoffs_*.nii.gz"):
            file.unlink()

    def test_save_to_nii_dtype_preservation(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test that dtype is correctly applied in NNLS saving."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        cutoffs = [(0.0, 0.01), (0.01, 0.1)]
        fit_results.apply_cutoffs(cutoffs)

        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        file_path = temp_dir / "test_dtype"

        fit_results.save_to_nii(
            file_path, mock_img, dtype=np.float32, separate_files=False
        )

        # Load and verify dtype
        loaded = RadImgArray(temp_dir / "test_dtype_d.nii.gz")
        assert loaded.dtype == np.float32

        # Cleanup
        for file in temp_dir.glob("test_dtype_*.nii.gz"):
            file.unlink()


# ========== Spectrum Tests ==========


class TestNNLSSpectrum:
    """Test NNLS spectrum handling and saving."""

    def test_spectrum_stored_after_eval(self, nnls_params, nnls_fit_results):
        """Test that spectrum is stored during eval_results."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        assert len(fit_results.spectrum) > 0
        for pixel in nnls_fit_results[3]:
            assert pixel in fit_results.spectrum
            assert isinstance(fit_results.spectrum[pixel], np.ndarray)

    def test_save_spectrum_to_excel_uses_correct_bins(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test that spectrum Excel export uses correct bins from params."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        file_path = temp_dir / "test_spectrum.xlsx"
        fit_results.save_spectrum_to_excel(file_path)

        assert file_path.exists()

        # Load and verify bins
        df = pd.read_excel(file_path)
        expected_bins = nnls_params.get_bins()

        # The DataFrame columns should include bins (as strings converted from floats)
        # The number of columns should be: pixel columns + bin columns
        assert df.shape[1] > len(expected_bins)  # Has pixel info + bins

        file_path.unlink()

    def test_save_spectrum_with_custom_bins(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test saving spectrum with custom bins."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        custom_bins = np.logspace(-4, -1, 350)
        file_path = temp_dir / "test_custom_bins.xlsx"

        fit_results.save_spectrum_to_excel(file_path, bins=custom_bins)

        assert file_path.exists()
        file_path.unlink()

    def test_spectrum_shape_matches_bins(self, nnls_params, nnls_fit_results):
        """Test that spectrum shape matches number of bins."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        n_bins = nnls_params.boundaries["n_bins"]

        for pixel, spectrum in fit_results.spectrum.items():
            assert len(spectrum) == n_bins


# # ========== Geometric Mean Peak Tests ==========


class TestGeometricMeanPeak:
    """Test the geometric mean peak calculation."""

    def test_geometric_mean_peak_basic(self):
        """Test basic geometric mean peak calculation."""
        positions = np.array([0.001, 0.002])
        heights = np.array([0.6, 0.4])

        mean_pos, total_height = NNLSResults.geometric_mean_peak(positions, heights)

        # Total height should be sum of heights
        assert np.isclose(total_height, 1.0)

        # Mean position should be between the two positions
        assert mean_pos > np.log10(positions[0])
        assert mean_pos < np.log10(positions[1])

    def test_geometric_mean_equal_weights(self):
        """Test geometric mean with equal weights."""
        positions = np.array([0.001, 0.01])
        heights = np.array([0.5, 0.5])

        mean_pos, total_height = NNLSResults.geometric_mean_peak(positions, heights)

        # With equal weights, should be geometric mean
        expected = np.log10(np.sqrt(positions[0] * positions[1]))
        assert np.isclose(mean_pos, expected)
        assert np.isclose(total_height, 1.0)

    def test_geometric_mean_single_peak(self):
        """Test geometric mean with effectively single peak (one weight is 0)."""
        positions = np.array([0.001])
        heights = np.array([1.0])

        mean_pos, total_height = NNLSResults.geometric_mean_peak(positions, heights)

        assert np.isclose(mean_pos, np.log10(positions[0]))
        assert np.isclose(total_height, 1.0)


# ========== Integration Tests ==========


class TestNNLSIntegration:
    """Integration tests for complete NNLS workflows."""

    def test_complete_workflow_eval_cutoff_save(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test complete workflow: evaluate -> apply cutoffs -> save."""
        fit_results = NNLSResults(nnls_params)

        # Step 1: Evaluate
        fit_results.eval_results(nnls_fit_results[0])
        assert len(fit_results.D) > 0

        # Step 2: Apply cutoffs
        cutoffs = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1)]
        fit_results.apply_cutoffs(cutoffs)
        assert fit_results.did_apply_cutoffs

        # Step 3: Save
        mock_img = RadImgArray(np.ones((11, 11, 11, 16)))
        nii_path = temp_dir / "workflow_test"
        excel_path = temp_dir / "workflow_test.xlsx"

        fit_results.save_to_nii(nii_path, mock_img, dtype=float, separate_files=False)
        fit_results.save_to_excel(excel_path)

        # Verify
        assert (temp_dir / "workflow_test_d.nii.gz").exists()
        assert excel_path.exists()

        # Cleanup
        for file in temp_dir.glob("workflow_test*"):
            file.unlink()

    def test_workflow_with_spectrum_saving(
        self, temp_dir, nnls_params, nnls_fit_results
    ):
        """Test workflow including spectrum saving."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        # Save both regular results and spectrum
        excel_results = temp_dir / "results.xlsx"
        excel_spectrum = temp_dir / "spectrum.xlsx"

        fit_results.save_to_excel(excel_results)
        fit_results.save_spectrum_to_excel(excel_spectrum)

        assert excel_results.exists()
        assert excel_spectrum.exists()

        # Cleanup
        excel_results.unlink()
        excel_spectrum.unlink()


# ========== Edge Cases ==========


class TestNNLSEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_results(self, nnls_params):
        """Test behavior with empty results."""
        fit_results = NNLSResults(nnls_params)

        # Don't evaluate any results
        assert len(fit_results.D) == 0
        assert len(fit_results.f) == 0
        assert not fit_results.did_apply_cutoffs

    def test_cutoffs_with_empty_results(self, nnls_params):
        """Test applying cutoffs to empty results."""
        fit_results = NNLSResults(nnls_params)

        cutoffs = [(0.0, 0.001), (0.001, 0.01)]
        fit_results.apply_cutoffs(cutoffs)

        # Should not crash, but did_apply_cutoffs should be True
        assert fit_results.did_apply_cutoffs
        assert len(fit_results.D) == 0

    def test_single_peak_results(self, nnls_params):
        """Test with spectrum containing only one peak."""
        fit_results = NNLSResults(nnls_params)

        spectrum = np.zeros(350)
        spectrum[50] = 1.0

        fit_results.eval_results([((0, 0, 0), spectrum)])

        # Should detect single peak
        assert len(fit_results.D[(0, 0, 0)]) == 1
        assert np.isclose(fit_results.f[(0, 0, 0)][0], 1.0)

    def test_very_narrow_cutoffs(self, nnls_params, nnls_fit_results):
        """Test with very narrow cutoff ranges."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        # Very narrow cutoffs
        cutoffs = [(0.0001, 0.0002), (0.005, 0.0051), (0.05, 0.051)]
        fit_results.apply_cutoffs(cutoffs)

        # Most ranges will likely have NaN
        for pixel in fit_results.D.keys():
            d_values = fit_results.D[pixel]
            # At least some should be NaN with such narrow ranges
            assert any(np.isnan(d_values))

    def test_cutoffs_extending_beyond_bins(self, nnls_params, nnls_fit_results):
        """Test cutoffs that extend beyond the bin range."""
        fit_results = NNLSResults(nnls_params)
        fit_results.eval_results(nnls_fit_results[0])

        bins = nnls_params.get_bins()
        # Cutoffs extending beyond bin range
        cutoffs = [(0.0, bins[0] * 0.5), (bins[-1] * 2, bins[-1] * 10)]

        fit_results.apply_cutoffs(cutoffs)

        # Should handle gracefully (likely all NaN)
        assert fit_results.did_apply_cutoffs
