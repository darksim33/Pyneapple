"""Tests for BaseFitter abstract base class.

BaseFitter is abstract, so a minimal concrete subclass (_ConcreteFitter) is
used throughout to exercise the shared initialisation and helper methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyneapple.fitters.base import BaseFitter
from pyneapple.models import MonoExpModel
from pyneapple.solvers import CurveFitSolver
from test_toolbox import B_VALUES


# ---------------------------------------------------------------------------
# Minimal concrete subclass — only exists to make BaseFitter instantiable
# ---------------------------------------------------------------------------


class _ConcreteFitter(BaseFitter):
    """Minimal BaseFitter subclass for unit testing."""

    def fit(self, xdata: np.ndarray, image: np.ndarray, **kwargs) -> "_ConcreteFitter":
        """Populates n_measurements and a dummy fitted_params_."""
        self.n_measurements = len(xdata)
        self.fitted_params_ = {"S0": np.ones(4)}
        return self

    def predict(self, xdata: np.ndarray, **kwargs) -> np.ndarray:
        """Returns zeros of the same shape as xdata."""
        return np.zeros_like(xdata)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def solver():
    """Minimal CurveFitSolver with MonoExp model."""
    return CurveFitSolver(
        model=MonoExpModel(),
        max_iter=250,
        tol=1e-8,
        p0={"S0": 1000.0, "D": 0.001},
        bounds={"S0": (0.0, 5000.0), "D": (1e-5, 0.1)},
    )


@pytest.fixture
def fitter(solver):
    """_ConcreteFitter backed by the minimal solver."""
    return _ConcreteFitter(solver=solver)


@pytest.fixture
def b_values() -> np.ndarray:
    """Standard 16-point b-value array."""
    return B_VALUES.copy()


# ---------------------------------------------------------------------------
# Init state
# ---------------------------------------------------------------------------


class TestBaseFitterInit:
    """Verify the initial state set up by BaseFitter.__init__."""

    @pytest.mark.unit
    def test_solver_stored(self, fitter, solver):
        """solver attribute holds the exact solver instance passed at construction."""
        assert fitter.solver is solver

    @pytest.mark.unit
    def test_fitted_params_is_empty_dict(self, fitter):
        """fitted_params_ is an empty dict before fit() is called."""
        assert isinstance(fitter.fitted_params_, dict)
        assert fitter.fitted_params_ == {}

    @pytest.mark.unit
    def test_results_is_none(self, fitter):
        """results_ is None before fit() is called."""
        assert fitter.results_ is None

    @pytest.mark.unit
    def test_pixel_indices_is_none(self, fitter):
        """pixel_indices is None before fit() is called."""
        assert fitter.pixel_indices is None

    @pytest.mark.unit
    def test_n_measurements_is_none(self, fitter):
        """n_measurements is None before fit() is called."""
        assert fitter.n_measurements is None

    @pytest.mark.unit
    def test_image_shape_is_none(self, fitter):
        """image_shape is None before fit() is called."""
        assert fitter.image_shape is None

    @pytest.mark.unit
    def test_extra_kwargs_stored(self, solver):
        """Extra fitter_kwargs passed at construction are stored."""
        f = _ConcreteFitter(solver=solver, custom_option=42)
        assert f.fitter_kwargs.get("custom_option") == 42


# ---------------------------------------------------------------------------
# get_fitted_params
# ---------------------------------------------------------------------------


class TestBaseFitterGetFittedParams:
    """Tests for get_fitted_params()."""

    @pytest.mark.unit
    def test_returns_empty_dict_before_fit(self, fitter):
        """get_fitted_params() returns an empty dict before fit()."""
        result = fitter.get_fitted_params()
        assert result == {}

    @pytest.mark.unit
    def test_returns_dict_after_fit(self, fitter, b_values):
        """get_fitted_params() returns a dict with parameter arrays after fit()."""
        image = np.ones((2, 2, 1, len(b_values)))
        fitter.fit(b_values, image)
        params = fitter.get_fitted_params()
        assert isinstance(params, dict)
        assert "S0" in params


# ---------------------------------------------------------------------------
# _extract_pixel_data
# ---------------------------------------------------------------------------


class TestExtractPixelData:
    """Tests for BaseFitter._extract_pixel_data()."""

    @pytest.mark.unit
    def test_raises_if_n_measurements_not_set(self, fitter):
        """_extract_pixel_data raises RuntimeError when n_measurements is None."""
        image = np.ones((3, 3, 1, 8))
        seg = np.ones((3, 3, 1), dtype=int)
        with pytest.raises(RuntimeError, match="n_measurements must be set"):
            fitter._extract_pixel_data(image, seg)

    @pytest.mark.unit
    def test_extracts_only_masked_pixels(self, fitter):
        """Only voxels where segmentation != 0 are returned."""
        n_b = 8
        fitter.n_measurements = n_b
        image = np.arange(4 * 4 * 1 * n_b, dtype=float).reshape(4, 4, 1, n_b)
        seg = np.zeros((4, 4, 1), dtype=int)
        seg[0, 0, 0] = 1
        seg[1, 1, 0] = 1
        pixels = fitter._extract_pixel_data(image, seg)
        assert pixels.shape == (2, n_b)

    @pytest.mark.unit
    def test_pixel_count_matches_nonzero_mask(self, fitter):
        """Extracted row count equals np.count_nonzero(segmentation)."""
        n_b = 5
        fitter.n_measurements = n_b
        image = np.ones((4, 4, 2, n_b))
        seg = np.zeros((4, 4, 2), dtype=int)
        seg[:2, :2, :] = 1  # 8 nonzero voxels
        pixels = fitter._extract_pixel_data(image, seg)
        assert pixels.shape[0] == int(np.count_nonzero(seg))

    @pytest.mark.unit
    def test_stores_pixel_indices(self, fitter):
        """pixel_indices is populated with (row, col, …) tuples after extraction."""
        n_b = 8
        fitter.n_measurements = n_b
        image = np.zeros((3, 3, 1, n_b))
        seg = np.zeros((3, 3, 1), dtype=int)
        seg[2, 1, 0] = 1
        fitter._extract_pixel_data(image, seg)
        assert fitter.pixel_indices is not None
        assert (2, 1, 0) in fitter.pixel_indices

    @pytest.mark.unit
    def test_pixel_indices_length_matches_nonzero_mask(self, fitter):
        """pixel_indices length equals the number of non-zero segmentation entries."""
        n_b = 4
        fitter.n_measurements = n_b
        image = np.ones((3, 3, 1, n_b))
        seg = np.zeros((3, 3, 1), dtype=int)
        seg[0, :, 0] = 1  # 3 voxels in first row
        fitter._extract_pixel_data(image, seg)
        assert len(fitter.pixel_indices) == 3


# ---------------------------------------------------------------------------
# _check_fitted
# ---------------------------------------------------------------------------


class TestBaseFitterCheckFitted:
    """Tests for _check_fitted() guard method."""

    @pytest.mark.unit
    def test_does_not_raise_when_params_populated(self, fitter):
        """_check_fitted does not raise when fitted_params_ is a non-None dict."""
        fitter.fitted_params_ = {"S0": np.array([1.0])}
        fitter._check_fitted()  # must not raise

    @pytest.mark.unit
    def test_raises_when_fitted_params_is_none(self, fitter):
        """_check_fitted raises RuntimeError when fitted_params_ is None."""
        fitter.fitted_params_ = None
        with pytest.raises(RuntimeError):
            fitter._check_fitted()

    @pytest.mark.unit
    def test_raises_when_fitted_params_is_empty_dict(self, fitter):
        """_check_fitted raises RuntimeError when fitted_params_ is an empty dict.

        Regression test for Bug 8: the original guard used ``is None`` which
        never fired because fitted_params_ is initialised as ``{}``, not None.
        """
        fitter.fitted_params_ = {}
        with pytest.raises(RuntimeError, match="has not been fitted"):
            fitter._check_fitted()
