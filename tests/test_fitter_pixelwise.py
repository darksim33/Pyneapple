"""Tests for PixelWiseFitter — initialisation, fitting, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from pyneapple.fitters import PixelWiseFitter
from pyneapple.models import MonoExpModel
from test_toolbox import B_VALUES, make_monoexp_image, make_monoexp_solver


# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

N_B = len(B_VALUES)

# Local alias so existing test calls remain unchanged
_make_image = make_monoexp_image


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
    """PixelWiseFitter backed by the monoexp solver."""
    return PixelWiseFitter(solver=monoexp_solver)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestPixelWiseFitterInit:
    """Verify the initial state of a freshly constructed PixelWiseFitter."""

    @pytest.mark.unit
    def test_solver_stored(self, fitter, monoexp_solver):
        """solver attribute holds the exact solver instance passed at construction."""
        assert fitter.solver is monoexp_solver

    @pytest.mark.unit
    def test_fitted_params_initially_empty(self, fitter):
        """fitted_params_ is an empty dict before fit() is called."""
        assert fitter.fitted_params_ == {}

    @pytest.mark.unit
    def test_pixel_indices_initially_none(self, fitter):
        """pixel_indices is None before fit() is called."""
        assert fitter.pixel_indices is None

    @pytest.mark.unit
    def test_n_measurements_initially_none(self, fitter):
        """n_measurements is None before fit() is called."""
        assert fitter.n_measurements is None


# ---------------------------------------------------------------------------
# fit() — return value and side-effects
# ---------------------------------------------------------------------------


class TestPixelWiseFitterFit:
    """Tests for the core fit() behaviour of PixelWiseFitter."""

    @pytest.mark.unit
    def test_fit_returns_self(self, fitter, b_values):
        """fit() returns self to enable method chaining."""
        result = fitter.fit(b_values, _make_image())
        assert result is fitter

    @pytest.mark.unit
    def test_fit_sets_n_measurements(self, fitter, b_values):
        """fit() sets n_measurements to the number of b-values."""
        fitter.fit(b_values, _make_image())
        assert fitter.n_measurements == N_B

    @pytest.mark.unit
    def test_fit_sets_image_shape(self, fitter, b_values):
        """fit() sets image_shape to the full 4-D image shape."""
        image = _make_image(n_x=4, n_y=4, n_z=1)
        fitter.fit(b_values, image)
        assert fitter.image_shape == image.shape

    @pytest.mark.unit
    def test_fit_populates_fitted_params_keys(self, fitter, b_values):
        """fit() populates fitted_params_ with keys matching the model's param_names."""
        fitter.fit(b_values, _make_image())
        assert set(fitter.fitted_params_.keys()) == {"S0", "D"}

    @pytest.mark.unit
    def test_fit_populates_pixel_indices(self, fitter, b_values):
        """fit() sets pixel_indices to a list with one entry per fitted voxel."""
        fitter.fit(b_values, _make_image(n_x=4, n_y=4, n_z=1))
        assert fitter.pixel_indices is not None
        assert len(fitter.pixel_indices) == 4 * 4 * 1

    @pytest.mark.unit
    def test_fitted_params_length_matches_pixel_count(self, fitter, b_values):
        """fitted_params_ arrays have one entry per fitted voxel."""
        image = _make_image(n_x=3, n_y=3, n_z=1)
        seg = np.zeros((3, 3, 1), dtype=int)
        seg[:2, :2, 0] = 1  # 4 ROI voxels
        fitter.fit(b_values, image, segmentation=seg)
        assert len(fitter.fitted_params_["S0"]) == 4

    @pytest.mark.unit
    def test_fit_with_segmentation_restricts_pixels(self, fitter, b_values):
        """Passing a partial segmentation fits only the masked voxels."""
        image = _make_image()  # (4, 4, 1, N_B)
        seg = np.zeros((4, 4, 1), dtype=int)
        seg[:2, :2, 0] = 1  # 4 voxels
        fitter.fit(b_values, image, segmentation=seg)
        assert len(fitter.pixel_indices) == 4

    @pytest.mark.unit
    def test_second_fit_overwrites_previous_results(self, fitter, b_values):
        """Calling fit() a second time replaces the previous fitted_params_."""
        fitter.fit(b_values, _make_image(S0=1000.0))
        first_s0 = fitter.fitted_params_["S0"].copy()
        fitter.fit(b_values, _make_image(S0=500.0))
        second_s0 = fitter.fitted_params_["S0"]
        assert not np.allclose(first_s0, second_s0, rtol=0.01), (
            "S0 values should differ after re-fitting with different signal amplitude"
        )

    @pytest.mark.unit
    def test_fit_recovers_true_params_no_segmentation(self, fitter, b_values):
        """fit() without segmentation recovers noise-free MonoExp parameters."""
        S0_true, D_true = 1000.0, 0.001
        fitter.fit(b_values, _make_image(n_x=2, n_y=2, n_z=1, S0=S0_true, D=D_true))
        params = fitter.get_fitted_params()
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)

    @pytest.mark.unit
    def test_fit_recovers_true_params_with_segmentation(self, fitter, b_values):
        """fit() with segmentation recovers noise-free MonoExp parameters in the ROI."""
        S0_true, D_true = 800.0, 0.0015
        image = _make_image(S0=S0_true, D=D_true)
        seg = np.zeros((4, 4, 1), dtype=int)
        seg[1:3, 1:3, 0] = 1
        fitter.fit(b_values, image, segmentation=seg)
        params = fitter.get_fitted_params()
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)

    @pytest.mark.unit
    def test_get_fitted_params_matches_fitted_params_(self, fitter, b_values):
        """get_fitted_params() returns the same content as fitted_params_."""
        fitter.fit(b_values, _make_image())
        assert fitter.get_fitted_params() is fitter.fitted_params_


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestPixelWiseFitterValidation:
    """Tests that fit() propagates validation errors correctly."""

    @pytest.mark.unit
    def test_raises_on_2d_xdata(self, fitter):
        """fit() raises ValueError when xdata is not 1-D."""
        with pytest.raises(ValueError, match="1D"):
            fitter.fit(np.ones((8, 2)), _make_image())

    @pytest.mark.unit
    def test_raises_on_xdata_length_mismatch(self, fitter):
        """fit() raises ValueError when xdata length doesn't match image last dim."""
        short_b = np.array([0, 50, 100], dtype=float)  # 3, shorter than image's N_B
        with pytest.raises(ValueError):
            fitter.fit(short_b, _make_image())

    @pytest.mark.unit
    def test_raises_on_wrong_segmentation_shape(self, fitter, b_values):
        """fit() raises ValueError when segmentation spatial shape doesn't match image."""
        image = _make_image()  # spatial (4, 4, 1)
        bad_seg = np.ones((3, 3, 1), dtype=int)  # wrong spatial shape
        with pytest.raises(ValueError):
            fitter.fit(b_values, image, segmentation=bad_seg)


# ---------------------------------------------------------------------------
# Fixed parameter maps
# ---------------------------------------------------------------------------


class TestPixelWiseFitterFixedParams:
    """Tests for fixed_param_maps support in PixelWiseFitter."""

    @pytest.fixture
    def t1_fitter(self):
        """Fitter with T1-enabled MonoExp model and T1 in p0/bounds."""
        from pyneapple.models import MonoExpModel
        from pyneapple.solvers import CurveFitSolver

        model = MonoExpModel(fit_t1=True, repetition_time=3000.0)
        solver = CurveFitSolver(
            model=model,
            max_iter=2000,
            tol=1e-10,
            p0={"S0": 900.0, "D": 0.001, "T1": 1000.0},
            bounds={"S0": (1.0, 5000.0), "D": (1e-5, 0.1), "T1": (100.0, 5000.0)},
        )
        return PixelWiseFitter(solver=solver)

    @staticmethod
    def _make_t1_image(n_x=4, n_y=4, n_z=1, S0=1000.0, D=0.001, T1=1000.0, TR=3000.0):
        """Create a noise-free 4-D T1-corrected monoexp image."""
        from pyneapple.models import MonoExpModel

        model = MonoExpModel(fit_t1=True, repetition_time=TR)
        signal = model.forward(B_VALUES, S0, D, T1)
        return np.tile(signal, (n_x, n_y, n_z, 1))

    @pytest.mark.unit
    def test_fixed_param_maps_recovers_params(self, t1_fitter, b_values):
        """Per-pixel T1 map yields correct S0 and D."""
        S0_true, D_true, T1_true = 1000.0, 0.001, 1000.0
        image = self._make_t1_image(n_x=2, n_y=2, S0=S0_true, D=D_true, T1=T1_true)
        t1_map = np.full((2, 2, 1), T1_true)
        t1_fitter.fit(b_values, image, fixed_param_maps={"T1": t1_map})
        params = t1_fitter.get_fitted_params()
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)

    @pytest.mark.unit
    def test_fixed_param_maps_wrong_shape_raises(self, t1_fitter, b_values):
        """ValueError when fixed_param_maps shape doesn't match image spatial shape."""
        image = self._make_t1_image(n_x=4, n_y=4)
        bad_map = np.ones((2, 2, 1))  # spatial mismatch
        with pytest.raises(ValueError, match="has shape"):
            t1_fitter.fit(b_values, image, fixed_param_maps={"T1": bad_map})

    @pytest.mark.unit
    def test_fixed_param_maps_unknown_name_raises(self, t1_fitter, b_values):
        """ValueError when fixed_param_maps key is not a valid param name."""
        image = self._make_t1_image(n_x=4, n_y=4)
        bad_map = np.ones((4, 4, 1))
        with pytest.raises(ValueError, match="Unknown fixed parameter"):
            t1_fitter.fit(b_values, image, fixed_param_maps={"BOGUS": bad_map})


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPixelWiseFitterPredict:
    """Tests for predict() on a fitted PixelWiseFitter."""

    @pytest.mark.unit
    def test_predict_runs_without_error(self, fitter, b_values):
        """predict() executes without raising after fit()."""
        fitter.fit(b_values, _make_image())
        predictions = fitter.predict(b_values)
        assert isinstance(predictions, np.ndarray)

    @pytest.mark.unit
    def test_predict_output_shape_matches_image(self, fitter, b_values):
        """predict() returns an array with the same spatial shape as the input image."""
        n_x, n_y, n_z = 3, 4, 2
        image = make_monoexp_image(n_x=n_x, n_y=n_y, n_z=n_z)
        fitter.fit(b_values, image)
        predictions = fitter.predict(b_values)
        expected_shape = (n_x, n_y, n_z, len(b_values))
        assert predictions.shape == expected_shape

    @pytest.mark.unit
    def test_predict_reconstructs_signal(self, fitter, b_values):
        """predict() reproduces the fitted signal from noise-free data."""
        S0_true, D_true = 1000.0, 0.001
        image = make_monoexp_image(S0=S0_true, D=D_true)
        fitter.fit(b_values, image)
        predictions = fitter.predict(b_values)
        expected = MonoExpModel().forward(b_values, S0_true, D_true)
        np.testing.assert_allclose(predictions[0, 0, 0], expected, rtol=1e-2)

    @pytest.mark.unit
    def test_predict_raises_on_2d_xdata(self, fitter, b_values):
        """predict() raises ValueError when xdata is not 1-D."""
        fitter.fit(b_values, _make_image())
        with pytest.raises(ValueError, match="1D"):
            fitter.predict(np.ones((8, 2)))
