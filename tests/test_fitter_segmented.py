"""Tests for SegmentedFitter — two-step sequential model fitting."""

from __future__ import annotations

import numpy as np
import pytest

from pyneapple.fitters import SegmentedFitter
from pyneapple.models import MonoExpModel, BiExpModel
from pyneapple.solvers import CurveFitSolver
from test_toolbox import B_VALUES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_B = len(B_VALUES)
TRUE_S0 = 1000.0
TRUE_D = 0.001
TRUE_F1 = 0.3
TRUE_D1 = 0.01  # fast component (perfusion)
TRUE_D2 = 0.001  # slow component (diffusion ~ ADC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_biexp_image(
    n_x: int = 4,
    n_y: int = 4,
    n_z: int = 1,
    f1: float = TRUE_F1,
    D1: float = TRUE_D1,
    D2: float = TRUE_D2,
    b_values: np.ndarray = B_VALUES,
) -> np.ndarray:
    """Create a noise-free 4-D biexponential DWI image (reduced mode)."""
    model = BiExpModel(fit_reduced=True)
    signal = model.forward(b_values, f1, D1, D2)
    return np.tile(signal, (n_x, n_y, n_z, 1))


def _make_monoexp_solver(**overrides) -> CurveFitSolver:
    """CurveFitSolver with MonoExpModel for Step 1."""
    kwargs = dict(
        model=MonoExpModel(),
        max_iter=250,
        tol=1e-8,
        p0={"S0": 1.0, "D": 0.001},
        bounds={"S0": (0.01, 5.0), "D": (1e-5, 0.1)},
    )
    kwargs.update(overrides)
    return CurveFitSolver(**kwargs)


def _make_biexp_solver(**overrides) -> CurveFitSolver:
    """CurveFitSolver with BiExpModel (reduced) for Step 2."""
    kwargs = dict(
        model=BiExpModel(fit_reduced=True),
        max_iter=500,
        tol=1e-8,
        p0={"f1": 0.3, "D1": 0.01, "D2": 0.001},
        bounds={"f1": (0.0, 1.0), "D1": (0.001, 0.1), "D2": (1e-5, 0.01)},
    )
    kwargs.update(overrides)
    return CurveFitSolver(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def b_values() -> np.ndarray:
    """Standard 16-point b-value array."""
    return B_VALUES.copy()


@pytest.fixture
def step1_solver() -> CurveFitSolver:
    """MonoExp solver for Step 1."""
    return _make_monoexp_solver()


@pytest.fixture
def step2_solver() -> CurveFitSolver:
    """BiExp solver for Step 2."""
    return _make_biexp_solver()


@pytest.fixture
def fitter(step1_solver, step2_solver) -> SegmentedFitter:
    """SegmentedFitter with D fixed from Step 1 mapping to D2."""
    return SegmentedFitter(
        step1_solver=step1_solver,
        step2_solver=step2_solver,
        step1_bvalue_range=(200, None),
        fixed_from_step1=["D"],
        param_mapping={"D": "D2"},
    )


@pytest.fixture
def fitter_no_fix(step1_solver, step2_solver) -> SegmentedFitter:
    """SegmentedFitter without any fixed params (two independent steps)."""
    return SegmentedFitter(
        step1_solver=step1_solver,
        step2_solver=step2_solver,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestSegmentedFitterInit:
    """Verify the initial state of a freshly constructed SegmentedFitter."""

    @pytest.mark.unit
    def test_solver_is_step2(self, fitter, step2_solver):
        """solver attribute references the Step 2 solver."""
        assert fitter.solver is step2_solver

    @pytest.mark.unit
    def test_step1_solver_stored(self, fitter, step1_solver):
        """step1_solver attribute holds the Step 1 solver."""
        assert fitter.step1_solver is step1_solver

    @pytest.mark.unit
    def test_fitted_params_initially_empty(self, fitter):
        """fitted_params_ is an empty dict before fit()."""
        assert fitter.fitted_params_ == {}

    @pytest.mark.unit
    def test_step1_params_initially_empty(self, fitter):
        """step1_params_ is an empty dict before fit()."""
        assert fitter.step1_params_ == {}

    @pytest.mark.unit
    def test_fixed_from_step1_stored(self, fitter):
        """fixed_from_step1 is stored correctly."""
        assert fitter.fixed_from_step1 == ["D"]

    @pytest.mark.unit
    def test_param_mapping_stored(self, fitter):
        """param_mapping is stored correctly."""
        assert fitter.param_mapping == {"D": "D2"}

    @pytest.mark.unit
    def test_bvalue_range_stored(self, fitter):
        """step1_bvalue_range is stored correctly."""
        assert fitter.step1_bvalue_range == (200, None)

    @pytest.mark.unit
    def test_no_fix_defaults(self, fitter_no_fix):
        """Without fixed params, defaults are empty."""
        assert fitter_no_fix.fixed_from_step1 == []
        assert fitter_no_fix.param_mapping == {}
        assert fitter_no_fix.step1_bvalue_range is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestSegmentedFitterValidation:
    """Tests for constructor and fit() validation."""

    @pytest.mark.unit
    def test_invalid_fixed_param_name_raises(self, step2_solver):
        """ValueError when fixed_from_step1 contains a name not in Step 1 model."""
        solver1 = _make_monoexp_solver()
        with pytest.raises(ValueError, match="not a parameter"):
            SegmentedFitter(
                step1_solver=solver1,
                step2_solver=step2_solver,
                fixed_from_step1=["BOGUS"],
            )

    @pytest.mark.unit
    def test_invalid_mapping_target_raises(self, step1_solver):
        """ValueError when mapped name is not in Step 2 model."""
        solver2 = _make_biexp_solver()
        with pytest.raises(ValueError, match="not a parameter"):
            SegmentedFitter(
                step1_solver=step1_solver,
                step2_solver=solver2,
                fixed_from_step1=["D"],
                param_mapping={"D": "NONEXISTENT"},
            )

    @pytest.mark.unit
    def test_empty_bvalue_range_raises(self, fitter, b_values):
        """ValueError when no b-values fall in the specified range."""
        fitter.step1_bvalue_range = (9999, None)
        with pytest.raises(ValueError, match="No b-values"):
            fitter.fit(b_values, _make_biexp_image())

    @pytest.mark.unit
    def test_fewer_than_3_bvalues_raises(self, step1_solver, step2_solver, b_values):
        """ValueError when fewer than 3 b-values fall in the Step 1 range."""
        f = SegmentedFitter(
            step1_solver=step1_solver,
            step2_solver=step2_solver,
            step1_bvalue_range=(1000, None),
        )
        with pytest.raises(ValueError, match="at least 3 b-values"):
            f.fit(b_values, _make_biexp_image())

    @pytest.mark.unit
    def test_raises_on_2d_xdata(self, fitter):
        """fit() raises ValueError when xdata is not 1-D."""
        with pytest.raises(ValueError, match="1D"):
            fitter.fit(np.ones((8, 2)), _make_biexp_image())


# ---------------------------------------------------------------------------
# B-value subsetting
# ---------------------------------------------------------------------------


class TestSegmentedFitterBvalueSubset:
    """Tests for the b-value subsetting logic."""

    @pytest.mark.unit
    def test_subset_lower_bound(self, fitter, b_values):
        """step1_bvalue_range=(200, None) keeps only b >= 200."""
        sub_xdata, sub_image = fitter._subset_bvalues(b_values, _make_biexp_image())
        assert np.all(sub_xdata >= 200)
        assert sub_image.shape[-1] == sub_xdata.shape[0]

    @pytest.mark.unit
    def test_subset_upper_bound(self, step1_solver, step2_solver, b_values):
        """step1_bvalue_range=(None, 500) keeps only b <= 500."""
        f = SegmentedFitter(
            step1_solver=step1_solver,
            step2_solver=step2_solver,
            step1_bvalue_range=(None, 500),
        )
        sub_xdata, _ = f._subset_bvalues(b_values, _make_biexp_image())
        assert np.all(sub_xdata <= 500)

    @pytest.mark.unit
    def test_subset_both_bounds(self, step1_solver, step2_solver, b_values):
        """step1_bvalue_range=(100, 800) keeps 100 <= b <= 800."""
        f = SegmentedFitter(
            step1_solver=step1_solver,
            step2_solver=step2_solver,
            step1_bvalue_range=(100, 800),
        )
        sub_xdata, _ = f._subset_bvalues(b_values, _make_biexp_image())
        assert np.all(sub_xdata >= 100)
        assert np.all(sub_xdata <= 800)

    @pytest.mark.unit
    def test_no_range_uses_all_bvalues(self, fitter_no_fix, b_values):
        """When step1_bvalue_range is None, all b-values are used."""
        sub_xdata, sub_image = fitter_no_fix._subset_bvalues(
            b_values, _make_biexp_image()
        )
        np.testing.assert_array_equal(sub_xdata, b_values)
        assert sub_image.shape[-1] == len(b_values)


# ---------------------------------------------------------------------------
# fit() — core behaviour
# ---------------------------------------------------------------------------


class TestSegmentedFitterFit:
    """Tests for the core two-step fit() pipeline."""

    @pytest.mark.unit
    def test_fit_returns_self(self, fitter, b_values):
        """fit() returns self for method chaining."""
        result = fitter.fit(b_values, _make_biexp_image())
        assert result is fitter

    @pytest.mark.unit
    def test_fit_populates_step1_params(self, fitter, b_values):
        """fit() populates step1_params_ with MonoExp parameter names."""
        fitter.fit(b_values, _make_biexp_image())
        assert "S0" in fitter.step1_params_
        assert "D" in fitter.step1_params_

    @pytest.mark.unit
    def test_fit_populates_fitted_params(self, fitter, b_values):
        """fit() populates fitted_params_ with Step 2 free params + fixed params."""
        fitter.fit(b_values, _make_biexp_image())
        # Step 2 free params: f1, D1 (D2 is fixed)
        assert "f1" in fitter.fitted_params_
        assert "D1" in fitter.fitted_params_
        # D2 merged back from Step 1
        assert "D2" in fitter.fitted_params_

    @pytest.mark.unit
    def test_fit_sets_pixel_indices(self, fitter, b_values):
        """fit() populates pixel_indices."""
        fitter.fit(b_values, _make_biexp_image(n_x=3, n_y=3, n_z=1))
        assert fitter.pixel_indices is not None
        assert len(fitter.pixel_indices) == 9

    @pytest.mark.unit
    def test_fit_sets_n_measurements(self, fitter, b_values):
        """fit() sets n_measurements to the full b-value count."""
        fitter.fit(b_values, _make_biexp_image())
        assert fitter.n_measurements == N_B

    @pytest.mark.unit
    def test_fit_no_fixed_params(self, fitter_no_fix, b_values):
        """Two independent steps without fixing any params."""
        fitter_no_fix.fit(b_values, _make_biexp_image())
        assert "f1" in fitter_no_fix.fitted_params_
        assert "D1" in fitter_no_fix.fitted_params_
        assert "D2" in fitter_no_fix.fitted_params_
        assert "S0" in fitter_no_fix.step1_params_
        assert "D" in fitter_no_fix.step1_params_

    @pytest.mark.unit
    def test_fit_with_segmentation(self, fitter, b_values):
        """Segmentation mask is honoured in both steps."""
        image = _make_biexp_image(n_x=4, n_y=4, n_z=1)
        seg = np.zeros((4, 4, 1), dtype=int)
        seg[:2, :2, 0] = 1  # 4 voxels only
        fitter.fit(b_values, image, segmentation=seg)
        assert len(fitter.pixel_indices) == 4
        for arr in fitter.fitted_params_.values():
            assert len(np.atleast_1d(arr)) == 4

    @pytest.mark.unit
    def test_second_fit_overwrites(self, fitter, b_values):
        """A second fit() replaces previous results."""
        fitter.fit(b_values, _make_biexp_image(f1=0.3))
        first_f1 = fitter.fitted_params_["f1"].copy()
        fitter.fit(b_values, _make_biexp_image(f1=0.6))
        second_f1 = fitter.fitted_params_["f1"]
        # With very different f1 ground truth, fitted values should differ
        assert not np.allclose(first_f1, second_f1, atol=0.05)


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------


class TestSegmentedFitterRecovery:
    """Tests that the segmented approach recovers known parameter values."""

    @pytest.mark.unit
    def test_step1_recovers_adc(self, fitter, b_values):
        """Step 1 recovers an ADC close to D2 (slow diffusion component)."""
        image = _make_biexp_image(D2=0.001)
        fitter.fit(b_values, image)
        # The monoexp ADC from high b-values should approximate D2
        adc = fitter.step1_params_["D"]
        np.testing.assert_allclose(adc, 0.001, atol=5e-4)

    @pytest.mark.unit
    def test_step2_recovers_f1(self, fitter, b_values):
        """Step 2 recovers the perfusion fraction f1."""
        image = _make_biexp_image(f1=0.3, D1=0.01, D2=0.001)
        fitter.fit(b_values, image)
        np.testing.assert_allclose(fitter.fitted_params_["f1"], 0.3, atol=0.1)

    @pytest.mark.unit
    def test_step2_recovers_d1(self, fitter, b_values):
        """Step 2 recovers the fast diffusion coefficient D1."""
        image = _make_biexp_image(f1=0.3, D1=0.01, D2=0.001)
        fitter.fit(b_values, image)
        np.testing.assert_allclose(fitter.fitted_params_["D1"], 0.01, atol=0.005)

    @pytest.mark.unit
    def test_d2_from_step1_matches(self, fitter, b_values):
        """D2 in fitted_params_ matches the ADC from step1_params_."""
        image = _make_biexp_image()
        fitter.fit(b_values, image)
        np.testing.assert_array_equal(
            fitter.fitted_params_["D2"], fitter.step1_params_["D"]
        )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestSegmentedFitterPredict:
    """Tests for predict() after fitting."""

    @pytest.mark.unit
    def test_predict_shape(self, fitter, b_values):
        """predict() returns an array matching the original image spatial shape."""
        image = _make_biexp_image(n_x=3, n_y=3, n_z=1)
        fitter.fit(b_values, image)
        pred = fitter.predict(b_values)
        assert pred.shape == image.shape

    @pytest.mark.unit
    def test_predict_before_fit_raises(self, fitter, b_values):
        """predict() raises RuntimeError if fit() hasn't been called."""
        with pytest.raises(RuntimeError, match="not been fitted"):
            fitter.predict(b_values)

    @pytest.mark.unit
    def test_predict_2d_xdata_raises(self, fitter, b_values):
        """predict() raises ValueError for 2D xdata."""
        fitter.fit(b_values, _make_biexp_image())
        with pytest.raises(ValueError, match="1D"):
            fitter.predict(np.ones((8, 2)))

    @pytest.mark.unit
    def test_predict_signal_non_negative(self, fitter, b_values):
        """Predicted signal is non-negative (physical constraint)."""
        fitter.fit(b_values, _make_biexp_image())
        pred = fitter.predict(b_values)
        assert np.all(pred >= 0)


# ---------------------------------------------------------------------------
# _reconstruct_volume (shared BaseFitter helper)
# ---------------------------------------------------------------------------


class TestReconstructVolume:
    """Tests for the _reconstruct_volume helper on BaseFitter."""

    @pytest.mark.unit
    def test_correct_shape(self, fitter):
        """Reconstructed volume has the requested spatial shape."""
        flat = np.array([1.0, 2.0, 3.0])
        indices = [(0, 0, 0), (1, 1, 0), (2, 2, 0)]
        vol = fitter._reconstruct_volume(flat, indices, (3, 3, 1))
        assert vol.shape == (3, 3, 1)

    @pytest.mark.unit
    def test_values_placed_correctly(self, fitter):
        """Values appear at the correct spatial positions."""
        flat = np.array([10.0, 20.0])
        indices = [(0, 1, 0), (2, 3, 0)]
        vol = fitter._reconstruct_volume(flat, indices, (4, 4, 1))
        assert vol[0, 1, 0] == 10.0
        assert vol[2, 3, 0] == 20.0

    @pytest.mark.unit
    def test_unfitted_voxels_are_zero(self, fitter):
        """Voxels not in pixel_indices are zero."""
        flat = np.array([5.0])
        indices = [(1, 1, 0)]
        vol = fitter._reconstruct_volume(flat, indices, (3, 3, 1))
        vol[1, 1, 0] = 0.0  # zero out the fitted voxel
        assert np.all(vol == 0)
