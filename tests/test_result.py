"""Tests for FitResult and _PixelFitResult."""

from __future__ import annotations

import numpy as np
import pytest

from pyneapple import FitResult
from pyneapple.solvers.base import _PixelFitResult
from pyneapple.fitters import PixelWiseFitter
from test_toolbox import B_VALUES, make_monoexp_image, make_monoexp_solver


# ---------------------------------------------------------------------------
# _PixelFitResult unit tests
# ---------------------------------------------------------------------------


class TestPixelFitResult:
    """Unit tests for the internal _PixelFitResult dataclass."""

    def test_required_field_only(self):
        """_PixelFitResult can be constructed with just params."""
        params = np.array([1000.0, 0.001])
        pr = _PixelFitResult(params=params)
        np.testing.assert_array_equal(pr.params, params)

    def test_defaults(self):
        """Optional fields default to expected sentinel values."""
        pr = _PixelFitResult(params=np.array([1.0]))
        assert pr.covariance is None
        assert pr.success is True
        assert pr.message is None
        assert pr.n_iterations is None
        assert pr.residual is None

    def test_success_false(self):
        """success=False is stored correctly."""
        pr = _PixelFitResult(params=np.zeros(2), success=False)
        assert pr.success is False

    def test_with_covariance(self):
        """covariance is stored and round-trips through the dataclass."""
        cov = np.eye(2) * 0.5
        pr = _PixelFitResult(params=np.ones(2), covariance=cov)
        np.testing.assert_array_equal(pr.covariance, cov)

    def test_with_message_and_n_iterations(self):
        """message and n_iterations are stored correctly."""
        pr = _PixelFitResult(
            params=np.ones(3),
            message="Converged",
            n_iterations=42,
        )
        assert pr.message == "Converged"
        assert pr.n_iterations == 42

    def test_with_residual(self):
        """residual is stored as a float."""
        pr = _PixelFitResult(params=np.ones(2), residual=0.123)
        assert pr.residual == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# FitResult construction unit tests
# ---------------------------------------------------------------------------


def _make_fit_result(
    n_pixels: int = 5,
    n_params: int = 2,
    all_success: bool = True,
    include_r_squared: bool = True,
) -> FitResult:
    """Helper: build a minimal valid FitResult for unit testing."""
    rng = np.random.default_rng(0)
    params = {
        "S0": rng.uniform(800, 1200, n_pixels),
        "D": rng.uniform(0.0008, 0.0012, n_pixels),
    }
    success = np.ones(n_pixels, dtype=bool)
    if not all_success:
        success[1] = False
    r_squared = rng.uniform(0.95, 1.0, n_pixels) if include_r_squared else None

    return FitResult(
        params=params,
        success=success,
        r_squared=r_squared,
        fit_time=0.25,
        n_pixels=n_pixels,
        solver_name="CurveFitSolver",
        model_name="MonoExpModel",
    )


class TestFitResultConstruction:
    """Tests for FitResult field storage and defaults."""

    def test_params_stored(self):
        """params dict is stored by reference."""
        fr = _make_fit_result()
        assert "S0" in fr.params
        assert "D" in fr.params

    def test_success_array_shape(self):
        """success has shape (n_pixels,)."""
        fr = _make_fit_result(n_pixels=7)
        assert fr.success.shape == (7,)
        assert fr.success.dtype == bool

    def test_fit_time_stored(self):
        """fit_time is stored as given."""
        fr = _make_fit_result()
        assert fr.fit_time == pytest.approx(0.25)

    def test_optional_fields_default_none(self):
        """Optional diagnostic fields default to None."""
        fr = FitResult(
            params={"S0": np.array([1000.0])},
            success=np.array([True]),
            n_pixels=1,
        )
        assert fr.n_iterations is None
        assert fr.messages is None
        assert fr.covariance is None
        assert fr.residuals is None
        assert fr.r_squared is None
        assert fr.image_shape is None
        assert fr.pixel_indices is None

    def test_solver_and_model_name(self):
        """solver_name and model_name are stored correctly."""
        fr = _make_fit_result()
        assert fr.solver_name == "CurveFitSolver"
        assert fr.model_name == "MonoExpModel"

    def test_n_iterations_stored(self):
        """n_iterations array is stored when provided."""
        iters = np.array([10, 12, 9])
        fr = FitResult(
            params={"S0": np.ones(3)},
            success=np.ones(3, dtype=bool),
            n_iterations=iters,
            n_pixels=3,
        )
        np.testing.assert_array_equal(fr.n_iterations, iters)

    def test_covariance_shape_stored(self):
        """covariance of shape (n_pixels, n_params, n_params) is stored."""
        cov = np.zeros((4, 2, 2))
        fr = FitResult(
            params={"S0": np.ones(4), "D": np.ones(4)},
            success=np.ones(4, dtype=bool),
            covariance=cov,
            n_pixels=4,
        )
        assert fr.covariance.shape == (4, 2, 2)

    def test_r_squared_range(self):
        """R² values stored in FitResult are in a reasonable range."""
        fr = _make_fit_result(include_r_squared=True)
        assert fr.r_squared is not None
        assert np.all(fr.r_squared <= 1.0 + 1e-9)
        assert np.all(fr.r_squared >= 0.0)


# ---------------------------------------------------------------------------
# FitResult convenience properties
# ---------------------------------------------------------------------------


class TestFitResultProperties:
    """Tests for n_converged, convergence_rate, and mean_r_squared."""

    def test_n_converged_all_success(self):
        """n_converged equals n_pixels when all pixels converged."""
        fr = _make_fit_result(n_pixels=5, all_success=True)
        assert fr.n_converged == 5

    def test_n_converged_one_failure(self):
        """n_converged is n_pixels - 1 when one pixel failed."""
        fr = _make_fit_result(n_pixels=5, all_success=False)
        assert fr.n_converged == 4

    def test_convergence_rate_all_success(self):
        """convergence_rate is 1.0 when all pixels converged."""
        fr = _make_fit_result(n_pixels=4, all_success=True)
        assert fr.convergence_rate == pytest.approx(1.0)

    def test_convergence_rate_one_failure(self):
        """convergence_rate is (n-1)/n when one pixel failed."""
        fr = _make_fit_result(n_pixels=5, all_success=False)
        assert fr.convergence_rate == pytest.approx(4 / 5)

    def test_convergence_rate_zero_pixels(self):
        """convergence_rate returns 0.0 when n_pixels is 0."""
        fr = FitResult(
            params={},
            success=np.array([], dtype=bool),
            n_pixels=0,
        )
        assert fr.convergence_rate == pytest.approx(0.0)

    def test_mean_r_squared_none_when_not_computed(self):
        """mean_r_squared is None when r_squared is None."""
        fr = _make_fit_result(include_r_squared=False)
        assert fr.mean_r_squared is None

    def test_mean_r_squared_value(self):
        """mean_r_squared is the nanmean of the r_squared array."""
        r2 = np.array([0.99, 0.98, 0.97, np.nan])
        fr = FitResult(
            params={"S0": np.ones(4)},
            success=np.ones(4, dtype=bool),
            r_squared=r2,
            n_pixels=4,
        )
        assert fr.mean_r_squared == pytest.approx(np.nanmean(r2))

    def test_mean_r_squared_all_nan(self):
        """mean_r_squared returns nan when all R² values are NaN."""
        r2 = np.full(3, np.nan)
        fr = FitResult(
            params={"S0": np.ones(3)},
            success=np.ones(3, dtype=bool),
            r_squared=r2,
            n_pixels=3,
        )
        assert np.isnan(fr.mean_r_squared)


# ---------------------------------------------------------------------------
# FitResult public export
# ---------------------------------------------------------------------------


class TestFitResultExport:
    """FitResult is accessible from the top-level pyneapple namespace."""

    def test_importable_from_pyneapple(self):
        """FitResult can be imported directly from pyneapple."""
        from pyneapple import FitResult as FR

        assert FR is FitResult


# ---------------------------------------------------------------------------
# Integration: fitter.results_ is populated after fit()
# ---------------------------------------------------------------------------


@pytest.fixture
def b_values():
    return B_VALUES.copy()


@pytest.fixture
def monoexp_image(b_values):
    """Small 2×2×1 noise-free MonoExp image."""
    return make_monoexp_image(n_x=2, n_y=2, n_z=1, b_values=b_values)


@pytest.fixture
def fitted_fitter(b_values, monoexp_image):
    """PixelWiseFitter after fitting a 2×2×1 monoexp image."""
    solver = make_monoexp_solver()
    fitter = PixelWiseFitter(solver=solver)
    fitter.fit(b_values, monoexp_image)
    return fitter


class TestFitResultIntegration:
    """Integration tests: FitResult produced by a real PixelWiseFitter.fit()."""

    def test_results_attribute_is_set(self, fitted_fitter):
        """fitter.results_ is a FitResult after fit()."""
        assert isinstance(fitted_fitter.results_, FitResult)

    def test_n_pixels_matches_image(self, fitted_fitter, monoexp_image):
        """n_pixels equals the total number of voxels in the image."""
        spatial_voxels = np.prod(monoexp_image.shape[:-1])
        assert fitted_fitter.results_.n_pixels == spatial_voxels

    def test_fit_time_is_positive(self, fitted_fitter):
        """fit_time is a positive float after a real fit."""
        assert fitted_fitter.results_.fit_time > 0.0

    def test_success_all_true_noise_free(self, fitted_fitter):
        """All pixels should converge on noise-free monoexp data."""
        assert np.all(fitted_fitter.results_.success)

    def test_n_converged_equals_n_pixels(self, fitted_fitter):
        """n_converged equals n_pixels for noise-free data."""
        fr = fitted_fitter.results_
        assert fr.n_converged == fr.n_pixels

    def test_convergence_rate_is_one(self, fitted_fitter):
        """convergence_rate is 1.0 for noise-free data."""
        assert fitted_fitter.results_.convergence_rate == pytest.approx(1.0)

    def test_r_squared_is_high(self, fitted_fitter):
        """R² is close to 1.0 for noise-free monoexp data."""
        r2 = fitted_fitter.results_.r_squared
        assert r2 is not None
        assert np.all(r2 > 0.999), f"Expected R²>0.999, got min={r2.min():.6f}"

    def test_params_keys_match_model(self, fitted_fitter):
        """params dict contains the model's parameter names."""
        fr = fitted_fitter.results_
        assert set(fr.params.keys()) == {"S0", "D"}

    def test_params_shape(self, fitted_fitter, monoexp_image):
        """Each param array has shape (n_pixels,)."""
        fr = fitted_fitter.results_
        n_pixels = np.prod(monoexp_image.shape[:-1])
        for name, arr in fr.params.items():
            assert arr.shape == (n_pixels,), f"Unexpected shape for {name}: {arr.shape}"

    def test_solver_name_stored(self, fitted_fitter):
        """solver_name reflects the actual solver class."""
        assert fitted_fitter.results_.solver_name == "CurveFitSolver"

    def test_model_name_stored(self, fitted_fitter):
        """model_name reflects the actual model class."""
        assert fitted_fitter.results_.model_name == "MonoExpModel"

    def test_covariance_shape(self, fitted_fitter, monoexp_image):
        """covariance has shape (n_pixels, n_params, n_params)."""
        fr = fitted_fitter.results_
        n_pixels = np.prod(monoexp_image.shape[:-1])
        n_params = 2  # S0, D
        assert fr.covariance is not None
        assert fr.covariance.shape == (n_pixels, n_params, n_params)

    def test_results_fit_returns_self(self, b_values, monoexp_image):
        """fit() returns self for method chaining."""
        solver = make_monoexp_solver()
        fitter = PixelWiseFitter(solver=solver)
        returned = fitter.fit(b_values, monoexp_image)
        assert returned is fitter
