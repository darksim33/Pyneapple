"""Tests for SegmentationWiseFitter — initialisation, fitting, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from pyneapple.models import MonoExpModel
from pyneapple.fitters.segmentationwise import SegmentationWiseFitter
from test_toolbox import B_VALUES, make_monoexp_image, make_monoexp_solver


# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

N_B = len(B_VALUES)

# Local alias so existing test calls remain unchanged
_make_image = make_monoexp_image


def _make_segmentation(n_x: int = 4, n_y: int = 4, n_z: int = 1) -> np.ndarray:
    """Create a segmentation with two labelled regions (1 and 2) and background (0).

    Region 1: top-left quadrant, Region 2: bottom-right quadrant, rest is background.
    """
    seg = np.zeros((n_x, n_y, n_z), dtype=int)
    half_x, half_y = n_x // 2, n_y // 2
    seg[:half_x, :half_y, :] = 1
    seg[half_x:, half_y:, :] = 2
    return seg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def b_values() -> np.ndarray:
    """Standard 16-point b-value array."""
    return B_VALUES.copy()


@pytest.fixture
def monoexp_solver():
    """CurveFitSolver with MonoExp model, reasonable p0 and bounds."""
    return make_monoexp_solver()


@pytest.fixture
def fitter(monoexp_solver):
    """SegmentationWiseFitter backed by the monoexp solver."""
    return SegmentationWiseFitter(solver=monoexp_solver)


@pytest.fixture
def segmentation():
    """Segmentation with two labelled regions and background."""
    return _make_segmentation()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestSegmentationWiseFitterInit:
    """Verify the initial state of a freshly constructed SegmentationWiseFitter."""

    @pytest.mark.unit
    def test_solver_stored(self, fitter, monoexp_solver):
        """solver attribute holds the exact solver instance passed at construction."""
        assert fitter.solver is monoexp_solver

    @pytest.mark.unit
    def test_fitted_params_initially_empty(self, fitter):
        """fitted_params_ is an empty dict before fit() is called."""
        assert fitter.fitted_params_ == {}

    @pytest.mark.unit
    def test_n_measurements_initially_none(self, fitter):
        """n_measurements is None before fit() is called."""
        assert fitter.n_measurements is None

    @pytest.mark.unit
    def test_image_shape_initially_none(self, fitter):
        """image_shape is None before fit() is called."""
        assert fitter.image_shape is None


# ---------------------------------------------------------------------------
# fit() — return value and side-effects
# ---------------------------------------------------------------------------


class TestSegmentationWiseFitterFit:
    """Tests for the core fit() behaviour of SegmentationWiseFitter."""

    @pytest.mark.unit
    def test_fit_returns_self(self, fitter, b_values, segmentation):
        """fit() returns self to enable method chaining."""
        result = fitter.fit(b_values, _make_image(), segmentation=segmentation)
        assert result is fitter

    @pytest.mark.unit
    def test_fit_sets_n_measurements(self, fitter, b_values, segmentation):
        """fit() sets n_measurements to the number of b-values."""
        fitter.fit(b_values, _make_image(), segmentation=segmentation)
        assert fitter.n_measurements == N_B

    @pytest.mark.unit
    def test_fit_sets_image_shape(self, fitter, b_values, segmentation):
        """fit() sets image_shape to the full 4-D image shape."""
        image = _make_image(n_x=4, n_y=4, n_z=1)
        fitter.fit(b_values, image, segmentation=segmentation)
        assert fitter.image_shape == image.shape

    @pytest.mark.unit
    def test_fit_populates_fitted_params_keys(self, fitter, b_values, segmentation):
        """fit() populates fitted_params_ with keys matching the model's param_names."""
        fitter.fit(b_values, _make_image(), segmentation=segmentation)
        assert set(fitter.fitted_params_.keys()) == {"S0", "D"}

    @pytest.mark.unit
    def test_fit_stores_segment_labels(self, fitter, b_values, segmentation):
        """fit() stores the unique segment labels found in the segmentation."""
        fitter.fit(b_values, _make_image(), segmentation=segmentation)
        np.testing.assert_array_equal(
            np.sort(fitter.segment_labels), np.array([0, 1, 2])
        )

    # TODO: add test for pixel_to_segment mapping once we have a way to access it without predict() or get_fitted_params()

    @pytest.mark.unit
    def test_fitted_params_length_matches_segment_count(
        self, fitter, b_values, segmentation
    ):
        """fitted_params_ arrays have one entry per segment (including background)."""
        fitter.fit(b_values, _make_image(), segmentation=segmentation)
        n_segments = len(np.unique(segmentation))
        assert len(fitter.fitted_params_["S0"]) == n_segments
        assert len(fitter.fitted_params_["D"]) == n_segments

    @pytest.mark.unit
    def test_second_fit_overwrites_previous_results(
        self, fitter, b_values, segmentation
    ):
        """Calling fit() a second time replaces the previous fitted_params_."""
        fitter.fit(b_values, _make_image(S0=1000.0), segmentation=segmentation)
        first_s0 = fitter.fitted_params_["S0"].copy()
        fitter.fit(b_values, _make_image(S0=500.0), segmentation=segmentation)
        second_s0 = fitter.fitted_params_["S0"]
        assert not np.allclose(
            first_s0, second_s0, rtol=0.01
        ), "S0 values should differ after re-fitting with different signal amplitude"

    @pytest.mark.unit
    def test_fit_recovers_true_params(self, fitter, b_values, segmentation):
        """fit() recovers noise-free MonoExp parameters for each segment."""
        S0_true, D_true = 1000.0, 0.001
        fitter.fit(
            b_values,
            _make_image(S0=S0_true, D=D_true),
            segmentation=segmentation,
        )
        params = fitter.get_fitted_params()
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)

    @pytest.mark.unit
    def test_get_fitted_params_matches_fitted_params_(
        self, fitter, b_values, segmentation
    ):
        """get_fitted_params() returns the same content as fitted_params_."""
        fitter.fit(b_values, _make_image(), segmentation=segmentation)
        assert fitter.get_fitted_params() is fitter.fitted_params_

    @pytest.mark.unit
    def test_fit_single_segment(self, fitter, b_values):
        """fit() works with a segmentation that has only one non-zero label."""
        seg = np.ones((4, 4, 1), dtype=int)
        S0_true, D_true = 800.0, 0.0015
        fitter.fit(b_values, _make_image(S0=S0_true, D=D_true), segmentation=seg)
        params = fitter.get_fitted_params()
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestSegmentationWiseFitterPredict:
    """Tests for predict() on a fitted SegmentationWiseFitter."""

    @pytest.mark.unit
    def test_predict_output_shape(self, fitter, b_values, segmentation):
        """predict() returns array of shape (*image.shape, n_measurements)."""
        image = _make_image()
        fitter.fit(b_values, image, segmentation=segmentation)
        predictions = fitter.predict(b_values)
        assert predictions.shape == (*image.shape[:-1], N_B)

    @pytest.mark.unit
    def test_predict_matches_original_signal(self, fitter, b_values, segmentation):
        """predict() on noise-free data reproduces the original signal closely."""
        S0_true, D_true = 1000.0, 0.001
        fitter.fit(
            b_values,
            _make_image(S0=S0_true, D=D_true),
            segmentation=segmentation,
        )
        predictions = fitter.predict(b_values)
        expected = MonoExpModel().forward(b_values, S0_true, D_true)
        # Every segment should reproduce the same signal (uniform image)
        for i in np.ndindex(predictions.shape[:-1]):
            np.testing.assert_allclose(predictions[i], expected, rtol=1e-2)

    @pytest.mark.unit
    def test_predict_raises_on_2d_xdata(self, fitter, b_values, segmentation):
        """predict() raises ValueError when xdata is not 1-D."""
        fitter.fit(b_values, _make_image(), segmentation=segmentation)
        with pytest.raises(ValueError, match="1D"):
            fitter.predict(np.ones((8, 2)))


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestSegmentationWiseFitterValidation:
    """Tests that fit() propagates validation errors correctly."""

    @pytest.mark.unit
    def test_raises_on_missing_segmentation(self, fitter, b_values):
        """fit() raises ValueError when segmentation is None."""
        with pytest.raises(ValueError, match="segmentation is required"):
            fitter.fit(b_values, _make_image())

    @pytest.mark.unit
    def test_raises_on_2d_xdata(self, fitter, segmentation):
        """fit() raises ValueError when xdata is not 1-D."""
        with pytest.raises(ValueError, match="1D"):
            fitter.fit(np.ones((8, 2)), _make_image(), segmentation=segmentation)

    @pytest.mark.unit
    def test_raises_on_xdata_length_mismatch(self, fitter, segmentation):
        """fit() raises ValueError when xdata length doesn't match image last dim."""
        short_b = np.array([0, 50, 100], dtype=float)  # 3, shorter than image's N_B
        with pytest.raises(ValueError):
            fitter.fit(short_b, _make_image(), segmentation=segmentation)

    @pytest.mark.unit
    def test_raises_on_wrong_segmentation_shape(self, fitter, b_values):
        """fit() raises ValueError when segmentation spatial shape doesn't match image."""
        image = _make_image()  # spatial (4, 4, 1)
        bad_seg = np.ones((3, 3, 1), dtype=int)  # wrong spatial shape
        with pytest.raises(ValueError):
            fitter.fit(b_values, image, segmentation=bad_seg)

    @pytest.mark.unit
    def test_predict_raises_before_fit(self, fitter, b_values):
        """predict() raises when called before fit()."""
        with pytest.raises((RuntimeError, KeyError)):
            fitter.predict(b_values)
