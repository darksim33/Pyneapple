"""Tests for utility/spectrum.py — NNLS spectrum post-processing functions."""

from __future__ import annotations

import numpy as np
import pytest

from pyneapple.utility.spectrum import (
    find_spectrum_peaks,
    calculate_peak_area,
    apply_cutoffs,
    geometric_mean_peak,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_spectrum(bins: np.ndarray, peaks: list[dict]) -> np.ndarray:
    """Build a synthetic spectrum as a sum of Gaussians.

    Parameters
    ----------
    bins : ndarray
        Bin positions (log-space D values).
    peaks : list[dict]
        Each entry has keys 'center', 'height', 'sigma' (in log10 units).
    """
    spec = np.zeros(len(bins))
    log_bins = np.log10(bins)
    for p in peaks:
        spec += p["height"] * np.exp(
            -0.5 * ((log_bins - np.log10(p["center"])) / p["sigma"]) ** 2
        )
    return spec


@pytest.fixture
def bins():
    """100-point log-spaced bins over [1e-4, 0.1]."""
    return np.logspace(-4, -1, 100)


@pytest.fixture
def single_peak_spectrum(bins):
    """Spectrum with one clear peak near D = 0.003."""
    return make_synthetic_spectrum(
        bins, [{"center": 0.003, "height": 1.0, "sigma": 0.15}]
    )


@pytest.fixture
def two_peak_spectrum(bins):
    """Spectrum with two clear peaks at D ≈ 0.001 and D ≈ 0.05."""
    return make_synthetic_spectrum(
        bins,
        [
            {"center": 0.001, "height": 0.6, "sigma": 0.12},
            {"center": 0.05, "height": 0.4, "sigma": 0.12},
        ],
    )


# ---------------------------------------------------------------------------
# find_spectrum_peaks
# ---------------------------------------------------------------------------


class TestFindSpectrumPeaks:
    """Tests for find_spectrum_peaks()."""

    def test_detects_single_peak(self, bins, single_peak_spectrum):
        """Finds one peak in a single-peak spectrum."""
        d_vals, f_vals = find_spectrum_peaks(single_peak_spectrum, bins)
        assert len(d_vals) == 1

    def test_peak_position_within_tolerance(self, bins, single_peak_spectrum):
        """Detected peak position is within 0.5 decade of the true position."""
        d_vals, _ = find_spectrum_peaks(single_peak_spectrum, bins)
        true_d = 0.003
        assert abs(np.log10(d_vals[0]) - np.log10(true_d)) < 0.5

    def test_fractions_sum_to_one(self, bins, two_peak_spectrum):
        """Fractions returned by find_spectrum_peaks sum to 1."""
        _, f_vals = find_spectrum_peaks(two_peak_spectrum, bins)
        assert f_vals.sum() == pytest.approx(1.0, abs=1e-5)

    def test_detects_two_peaks(self, bins, two_peak_spectrum):
        """Finds two peaks in a two-peak spectrum."""
        d_vals, f_vals = find_spectrum_peaks(two_peak_spectrum, bins)
        assert len(d_vals) == 2

    def test_returns_empty_arrays_for_flat_spectrum(self, bins):
        """Returns empty arrays when no peaks exceed height threshold."""
        flat = np.ones(len(bins)) * 0.05
        d_vals, f_vals = find_spectrum_peaks(flat, bins, height=0.1)
        assert len(d_vals) == 0 and len(f_vals) == 0

    def test_regularized_uses_area_fractions(self, bins, two_peak_spectrum):
        """Regularized mode returns non-negative normalised fractions."""
        _, f_reg = find_spectrum_peaks(two_peak_spectrum, bins, regularized=True)
        assert all(f >= 0 for f in f_reg)
        assert f_reg.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# calculate_peak_area
# ---------------------------------------------------------------------------


class TestCalculatePeakArea:
    """Tests for calculate_peak_area()."""

    def test_returns_one_area_per_peak(self, bins, two_peak_spectrum):
        """Returns the same number of areas as peaks."""
        from scipy import signal

        peaks, props = signal.find_peaks(two_peak_spectrum, height=0.1)
        areas = calculate_peak_area(two_peak_spectrum, peaks, props["peak_heights"])
        assert len(areas) == len(peaks)

    def test_areas_are_positive(self, bins, two_peak_spectrum):
        """All area values are positive."""
        from scipy import signal

        peaks, props = signal.find_peaks(two_peak_spectrum, height=0.1)
        areas = calculate_peak_area(two_peak_spectrum, peaks, props["peak_heights"])
        assert all(a > 0 for a in areas)


# ---------------------------------------------------------------------------
# geometric_mean_peak
# ---------------------------------------------------------------------------


class TestGeometricMeanPeak:
    """Tests for geometric_mean_peak()."""

    def test_single_peak_returns_log10_position(self):
        """Single-peak input returns log10(position) unchanged."""
        pos, height = geometric_mean_peak(np.array([0.01]), np.array([1.0]))
        assert pos == pytest.approx(np.log10(0.01), rel=1e-5)
        assert height == pytest.approx(1.0)

    def test_equal_weights_return_geometric_mean(self):
        """Equal-weight peaks return the geometric mean of positions."""
        positions = np.array([0.001, 0.01])
        heights = np.array([0.5, 0.5])
        pos, _ = geometric_mean_peak(positions, heights)
        expected = np.log10(np.sqrt(0.001 * 0.01))
        assert pos == pytest.approx(expected, rel=1e-5)

    def test_total_height_is_sum(self):
        """total_height equals the sum of input heights."""
        heights = np.array([0.3, 0.7])
        _, total = geometric_mean_peak(np.array([0.001, 0.01]), heights)
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_cutoffs
# ---------------------------------------------------------------------------


class TestApplyCutoffs:
    """Tests for apply_cutoffs()."""

    def test_one_peak_per_range(self):
        """Single peak per range is returned unchanged (position-wise)."""
        d = np.array([0.001, 0.05])
        f = np.array([0.4, 0.6])
        cutoffs = [(1e-4, 5e-3), (5e-3, 0.1)]
        d_new, f_new = apply_cutoffs(d, f, cutoffs)
        assert len(d_new) == 2
        assert d_new[0] == pytest.approx(0.001)
        assert d_new[1] == pytest.approx(0.05)

    def test_missing_peak_inserts_nan(self):
        """Range with no matching peak produces nan entry."""
        d = np.array([0.001])
        f = np.array([1.0])
        cutoffs = [(1e-4, 5e-3), (5e-3, 0.1)]  # second range has no peak
        d_new, f_new = apply_cutoffs(d, f, cutoffs)
        assert np.isnan(d_new[1])
        assert np.isnan(f_new[1])

    def test_multiple_peaks_merged(self):
        """Multiple peaks in one range are merged into a single entry."""
        d = np.array([0.002, 0.003])
        f = np.array([0.5, 0.5])
        cutoffs = [(1e-4, 5e-3)]
        d_new, f_new = apply_cutoffs(d, f, cutoffs)
        assert len(d_new) == 1
        assert not np.isnan(d_new[0])

    def test_fractions_normalised_ignoring_nan(self):
        """Fractions sum to 1 ignoring nan entries."""
        d = np.array([0.001])
        f = np.array([1.0])
        cutoffs = [(1e-4, 5e-3), (5e-3, 0.1)]
        _, f_new = apply_cutoffs(d, f, cutoffs)
        valid = f_new[~np.isnan(f_new)]
        assert valid.sum() == pytest.approx(1.0, abs=1e-5)

    def test_output_length_equals_n_cutoffs(self):
        """Output arrays have the same length as the cutoffs list."""
        cutoffs = [(1e-4, 5e-3), (5e-3, 0.1), (0.1, 1.0)]
        d = np.array([0.001, 0.05])
        f = np.array([0.5, 0.5])
        d_new, f_new = apply_cutoffs(d, f, cutoffs)
        assert len(d_new) == 3
        assert len(f_new) == 3
