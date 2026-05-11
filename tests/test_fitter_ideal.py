"""Tests for IDEALFitter — initialisation, input validation, resampling, and fitting."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from pyneapple.fitters.ideal import IDEALFitter, _BLOCK_AVERAGE
from test_toolbox import (
    B_VALUES,
    make_dim_steps,
    make_monoexp_image,
    make_monoexp_solver,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

N_B = len(B_VALUES)
_FULL_SHAPE = (4, 4)  # spatial shape used for most tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def b_values() -> np.ndarray:
    """Standard b-value array."""
    return B_VALUES.copy()


@pytest.fixture
def solver():
    """CurveFitSolver backed by MonoExpModel."""
    return make_monoexp_solver()


@pytest.fixture
def dim_steps() -> np.ndarray:
    """Two-step dim_steps for a 4×4 spatial grid: [[2,2],[4,4]]."""
    return make_dim_steps(_FULL_SHAPE)


@pytest.fixture
def step_tol() -> dict[str, float]:
    """Step tolerance matching MonoExpModel's 2 parameters (S0, D)."""
    return {"S0": 0.5, "D": 0.5}


@pytest.fixture
def fitter(solver, dim_steps, step_tol) -> IDEALFitter:
    """IDEALFitter ready for use in most tests (default downsampling_method)."""
    return IDEALFitter(solver=solver, dim_steps=dim_steps, step_tol=step_tol)


@pytest.fixture
def fitter_cv2(solver, dim_steps, step_tol) -> IDEALFitter:
    """IDEALFitter using cv2-based downsampling for comparison tests."""
    return IDEALFitter(
        solver=solver,
        dim_steps=dim_steps,
        step_tol=step_tol,
        downsampling_method="area",
    )


@pytest.fixture
def image_4d() -> np.ndarray:
    """Noise-free 4-D monoexp image of shape (4, 4, 1, N_B)."""
    return make_monoexp_image(n_x=4, n_y=4, n_z=1)


# ---------------------------------------------------------------------------
# TestIDEALFitterInit
# ---------------------------------------------------------------------------


class TestIDEALFitterInit:
    """Verify the initial state of a freshly constructed IDEALFitter."""

    @pytest.mark.unit
    def test_solver_stored(self, fitter, solver):
        """solver attribute holds the exact solver instance passed at construction."""
        assert fitter.solver is solver

    @pytest.mark.unit
    def test_dim_steps_stored(self, fitter, dim_steps):
        """dim_steps attribute holds the array passed at construction."""
        np.testing.assert_array_equal(fitter.dim_steps, dim_steps)

    @pytest.mark.unit
    def test_step_tol_stored(self, fitter, step_tol):
        """step_tol attribute holds the value passed at construction."""
        assert fitter.step_tol == step_tol

    @pytest.mark.unit
    def test_ideal_dims_default(self, solver, dim_steps, step_tol):
        """ideal_dims defaults to 2 when not specified."""
        f = IDEALFitter(solver=solver, dim_steps=dim_steps, step_tol=step_tol)
        assert f.ideal_dims == 2

    @pytest.mark.unit
    def test_segmentation_threshold_default(self, fitter):
        """segmentation_threshold defaults to 0.2."""
        assert fitter.segmentation_threshold == pytest.approx(0.2)

    @pytest.mark.unit
    def test_step_params_initially_empty(self, fitter):
        """step_params is an empty list before fit() is called."""
        assert fitter.step_params == []

    @pytest.mark.unit
    def test_fitted_params_initially_empty_dict(self, fitter):
        """fitted_params_ is an empty dict before fit() is called."""
        assert fitter.fitted_params_ == {}

    @pytest.mark.unit
    def test_downsampling_method_default_block_average(self, fitter):
        """downsampling_method defaults to the block_average sentinel."""
        assert fitter.downsampling_method == _BLOCK_AVERAGE

    @pytest.mark.unit
    def test_upsampling_method_default_cubic(self, fitter):
        """upsampling_method defaults to cv2.INTER_CUBIC."""
        assert fitter.upsampling_method == cv2.INTER_CUBIC

    @pytest.mark.unit
    def test_downsampling_method_linear(self, solver, dim_steps, step_tol):
        """downsampling_method stores cv2.INTER_LINEAR when 'linear' is requested."""
        f = IDEALFitter(
            solver=solver,
            dim_steps=dim_steps,
            step_tol=step_tol,
            downsampling_method="linear",
        )
        assert f.downsampling_method == cv2.INTER_LINEAR

    @pytest.mark.unit
    def test_downsampling_method_area(self, solver, dim_steps, step_tol):
        """downsampling_method stores cv2.INTER_AREA when 'area' is requested."""
        f = IDEALFitter(
            solver=solver,
            dim_steps=dim_steps,
            step_tol=step_tol,
            downsampling_method="area",
        )
        assert f.downsampling_method == cv2.INTER_AREA

    @pytest.mark.unit
    def test_upsampling_method_linear(self, solver, dim_steps, step_tol):
        """upsampling_method stores cv2.INTER_LINEAR when 'linear' is requested."""
        f = IDEALFitter(
            solver=solver,
            dim_steps=dim_steps,
            step_tol=step_tol,
            upsampling_method="linear",
        )
        assert f.upsampling_method == cv2.INTER_LINEAR

    @pytest.mark.unit
    def test_invalid_downsampling_method_raises(self, solver, dim_steps, step_tol):
        """Constructing with an unknown downsampling method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid downsampling method"):
            IDEALFitter(
                solver=solver,
                dim_steps=dim_steps,
                step_tol=step_tol,
                downsampling_method="nearest",
            )

    @pytest.mark.unit
    def test_invalid_upsampling_method_raises(self, solver, dim_steps, step_tol):
        """Constructing with an unknown upsampling method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid upsampling method"):
            IDEALFitter(
                solver=solver,
                dim_steps=dim_steps,
                step_tol=step_tol,
                upsampling_method="nearest",
            )

    @pytest.mark.unit
    def test_block_average_invalid_for_upsampling_raises(
        self, solver, dim_steps, step_tol
    ):
        """Passing 'block_average' as upsampling_method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid upsampling method"):
            IDEALFitter(
                solver=solver,
                dim_steps=dim_steps,
                step_tol=step_tol,
                upsampling_method="block_average",
            )


# ---------------------------------------------------------------------------
# TestIDEALFitterInputValidation
# ---------------------------------------------------------------------------


class TestIDEALFitterInputValidation:
    """Tests for the validation methods called during fit()."""

    # --- _validate_fitter_inputs ---

    @pytest.mark.unit
    def test_validate_fitter_inputs_non_2d_raises(self, solver, step_tol):
        """_validate_fitter_inputs raises ValueError when dim_steps is 1-D."""
        bad_dim_steps = np.array([2, 4])  # 1-D, not 2-D
        f = IDEALFitter(solver=solver, dim_steps=bad_dim_steps, step_tol=step_tol)
        with pytest.raises(ValueError, match="2D array"):
            f._validate_fitter_inputs(bad_dim_steps, ideal_dims=2)

    @pytest.mark.unit
    def test_validate_fitter_inputs_wrong_column_count_raises(self, solver, step_tol):
        """_validate_fitter_inputs raises ValueError when dim_steps has wrong number of columns."""
        bad_dim_steps = np.array(
            [[2, 2, 2], [2, 2, 2], [1, 1, 1]]
        )  # 3 rows (n_steps), should have 2 columns (ideal_dims=2)
        f = IDEALFitter(solver=solver, dim_steps=bad_dim_steps, step_tol=step_tol)
        with pytest.raises(ValueError, match="2 columns"):
            f._validate_fitter_inputs(bad_dim_steps, ideal_dims=2)

    @pytest.mark.unit
    def test_validate_fitter_inputs_non_monotonic_raises(self, solver, step_tol):
        """_validate_fitter_inputs raises ValueError when dim_steps is non-monotonic."""
        bad_dim_steps = np.array(
            [[4, 4], [2, 2]]
        )  # second row less than first (decreases)
        f = IDEALFitter(solver=solver, dim_steps=bad_dim_steps, step_tol=step_tol)
        with pytest.raises(ValueError, match="row 1"):
            f._validate_fitter_inputs(bad_dim_steps, ideal_dims=2)

    # --- _validate_step_tol ---

    @pytest.mark.unit
    def test_validate_step_tol_wrong_keys_raises(self, solver, dim_steps):
        """_validate_step_tol raises ValueError when step_tol keys do not match param_names."""
        # MonoExpModel has params S0 and D; "F" is not a valid key
        bad_tol = {"S0": 0.5, "F": 0.5}
        f = IDEALFitter(solver=solver, dim_steps=dim_steps, step_tol=bad_tol)
        with pytest.raises(ValueError, match="step_tol"):
            f._validate_step_tol()

    @pytest.mark.unit
    def test_validate_step_tol_not_dict_raises(self, solver, dim_steps):
        """_validate_step_tol raises ValueError when step_tol is not a dict."""
        bad_tol = [0.5, 0.5]
        f = IDEALFitter(solver=solver, dim_steps=dim_steps, step_tol=bad_tol)
        with pytest.raises(ValueError, match="dict"):
            f._validate_step_tol()

    # --- _validate_image_dims ---

    @pytest.mark.unit
    def test_validate_image_dims_2d_raises(self, fitter):
        """_validate_image_dims raises ValueError for a 2-D array."""
        with pytest.raises(ValueError, match="3 or 4"):
            fitter._validate_image_dims(np.ones((4, 8)))

    @pytest.mark.unit
    def test_validate_image_dims_3d_with_ideal3_raises(
        self, solver, dim_steps, step_tol
    ):
        """_validate_image_dims raises ValueError for 3-D image when ideal_dims=3."""
        f = IDEALFitter(
            solver=solver, dim_steps=dim_steps, step_tol=step_tol, ideal_dims=3
        )
        with pytest.raises(ValueError, match="not sufficient"):
            f._validate_image_dims(np.ones((4, 4, 8)))

    @pytest.mark.unit
    def test_validate_image_dims_3d_returns_4d(self, fitter):
        """_validate_image_dims expands a 3-D image to 4-D and returns it."""
        image_3d = np.ones((4, 4, 8))
        result = fitter._validate_image_dims(image_3d)
        assert result.ndim == 4, "Expected a 4-D array after expansion"
        assert result.shape == (4, 4, 1, 8)

    @pytest.mark.unit
    def test_validate_image_dims_4d_returns_unchanged(self, fitter):
        """_validate_image_dims returns a 4-D image without modification."""
        image_4d = np.ones((4, 4, 1, 8))
        result = fitter._validate_image_dims(image_4d)
        assert result.shape == image_4d.shape
        assert result is image_4d

    # --- fit() guards ---

    @pytest.mark.unit
    def test_fit_raises_on_2d_xdata(self, fitter, image_4d):
        """fit() raises ValueError when xdata is not 1-D."""
        with pytest.raises(ValueError, match="1D"):
            fitter.fit(np.ones((4, 2)), image_4d)

    @pytest.mark.unit
    def test_fit_raises_on_dim_steps_last_step_mismatch(
        self, solver, step_tol, b_values
    ):
        """fit() raises ValueError when last dim_steps step does not match image spatial dims."""
        wrong_dim_steps = np.array([[2, 2], [6, 6]])  # last step 6×6, image is 4×4
        f = IDEALFitter(solver=solver, dim_steps=wrong_dim_steps, step_tol=step_tol)
        image = make_monoexp_image(n_x=4, n_y=4, n_z=1)
        with pytest.raises(ValueError, match="last step"):
            f.fit(b_values, image)

    @pytest.mark.unit
    def test_fit_raises_on_segmentation_mismatch(self, solver, step_tol, b_values):
        """fit() raises ValueError when segmentation shape doesn't match image spatial shape."""
        dim_steps = np.array([[2, 2], [4, 4]])
        f = IDEALFitter(solver=solver, dim_steps=dim_steps, step_tol=step_tol)
        image = make_monoexp_image(n_x=4, n_y=4, n_z=1)
        segmentation = np.ones((5, 5, 1), dtype=int)  # Mismatched: 5×5 vs 4×4
        with pytest.raises(ValueError, match="(?i)segmentation"):
            f.fit(b_values, image, segmentation)


# ---------------------------------------------------------------------------
# TestIDEALFitterResamplingArrays
# ---------------------------------------------------------------------------


class TestIDEALFitterResamplingArrays:
    """Tests for the _downsampling_array and _upsampling_array helpers."""

    @pytest.mark.unit
    def test_downsampling_array_output_shape(self, fitter_cv2):
        """_downsampling_array returns an array with the expected spatial + last-dim shape."""
        array = np.random.rand(4, 4, 1, 8).astype(np.float32)
        target_shape = (2, 2, 1)
        result = fitter_cv2._downsampling_array(array, target_shape)
        assert result.shape == (2, 2, 1, 8)

    @pytest.mark.unit
    def test_downsampling_array_identity_same_size(self, fitter_cv2):
        """_downsampling_array with an equal target shape preserves values approximately."""
        array = np.random.rand(4, 4, 1, 3).astype(np.float32)
        result = fitter_cv2._downsampling_array(array, (4, 4, 1))
        np.testing.assert_allclose(result, array, rtol=1e-4)

    @pytest.mark.unit
    def test_downsampling_array_reduces_spatial_dims(self, fitter_cv2):
        """_downsampling_array spatial dims decrease when target is smaller than source."""
        array = np.ones((8, 8, 1, 4)).astype(np.float32)
        result = fitter_cv2._downsampling_array(array, (4, 4, 1))
        assert result.shape[:2] == (4, 4)

    @pytest.mark.unit
    def test_downsampling_array_accepts_numpy_target_shape(self, fitter_cv2):
        """_downsampling_array works when target_shape is a NumPy array (not just a plain tuple)."""
        array = np.ones((4, 4, 1, 8)).astype(np.float32)
        target_shape_np = np.array([2, 2, 1])
        result = fitter_cv2._downsampling_array(array, target_shape_np)
        assert result.shape == (2, 2, 1, 8)

    @pytest.mark.unit
    def test_downsampling_array_int_dtype_does_not_crash(self, fitter_cv2):
        """_downsampling_array handles integer dtype arrays without crashing."""
        int_array = np.ones((4, 4, 1, 1), dtype=int)
        result = fitter_cv2._downsampling_array(int_array, (2, 2, 1))
        assert result.shape == (2, 2, 1, 1)
        assert result.dtype.kind == "f", (
            "Output should be float after int→float conversion"
        )

    @pytest.mark.unit
    def test_upsampling_array_output_shape(self, fitter):
        """_upsampling_array returns an array with the expected spatial + last-dim shape."""
        array = np.random.rand(2, 2, 1, 8).astype(np.float32)
        result = fitter._upsampling_array(array, (4, 4, 1))
        assert result.shape == (4, 4, 1, 8)

    @pytest.mark.unit
    def test_upsampling_array_increases_spatial_dims(self, fitter):
        """_upsampling_array spatial dims increase when target is larger than source."""
        array = np.ones((2, 2, 1, 4)).astype(np.float32)
        result = fitter._upsampling_array(array, (4, 4, 1))
        assert result.shape[:2] == (4, 4)

    @pytest.mark.unit
    def test_upsampling_array_accepts_numpy_target_shape(self, fitter):
        """_upsampling_array works when target_shape is a NumPy array."""
        array = np.ones((2, 2, 1, 8)).astype(np.float32)
        target_shape_np = np.array([4, 4, 1])
        result = fitter._upsampling_array(array, target_shape_np)
        assert result.shape == (4, 4, 1, 8)

    @pytest.mark.unit
    def test_upsampling_array_int_dtype_does_not_crash(self, fitter):
        """_upsampling_array handles integer dtype arrays without crashing."""
        int_array = np.ones((2, 2, 1, 1), dtype=int)
        result = fitter._upsampling_array(int_array, (4, 4, 1))
        assert result.shape == (4, 4, 1, 1)
        assert result.dtype.kind == "f"


# ---------------------------------------------------------------------------
# TestIDEALFitterBlockAverage
# ---------------------------------------------------------------------------


class TestIDEALFitterBlockAverage:
    """Tests for the _block_average_array helper."""

    @pytest.mark.unit
    def test_block_average_output_shape(self, fitter):
        """_block_average_array returns an array with shape (tx, ty, tz, C)."""
        array = np.random.rand(4, 4, 1, 8).astype(np.float64)
        result = fitter._block_average_array(array, (2, 2, 1))
        assert result.shape == (2, 2, 1, 8)

    @pytest.mark.unit
    def test_block_average_uniform_image_preserves_value(self, fitter):
        """Block-averaging a uniform image returns the same constant value."""
        value = 3.14
        array = np.full((8, 8, 1, 4), value, dtype=np.float64)
        result = fitter._block_average_array(array, (4, 4, 1))
        np.testing.assert_allclose(result, value, rtol=1e-10)

    @pytest.mark.unit
    def test_block_average_known_values(self, fitter):
        """2×2 → 1×1: the single output bin equals the mean of all four pixels."""
        # Build a 2×2×1×1 image with known pixel values
        array = np.zeros((2, 2, 1, 1), dtype=np.float64)
        array[0, 0, 0, 0] = 1.0
        array[0, 1, 0, 0] = 2.0
        array[1, 0, 0, 0] = 3.0
        array[1, 1, 0, 0] = 4.0
        result = fitter._block_average_array(array, (1, 1, 1))
        expected = 2.5  # (1+2+3+4)/4
        np.testing.assert_allclose(result[0, 0, 0, 0], expected)

    @pytest.mark.unit
    def test_block_average_mask_excludes_pixels(self, fitter):
        """Pixels outside the mask are excluded from the bin mean."""
        # 2×2 image: pixel (1,1) is masked out; the remaining three average to 2.0
        array = np.zeros((2, 2, 1, 1), dtype=np.float64)
        array[0, 0, 0, 0] = 1.0
        array[0, 1, 0, 0] = 2.0
        array[1, 0, 0, 0] = 3.0
        array[1, 1, 0, 0] = 99.0  # masked — should not contribute

        mask = np.ones((2, 2, 1), dtype=bool)
        mask[1, 1, 0] = False

        result = fitter._block_average_array(array, (1, 1, 1), mask=mask)
        expected = (1.0 + 2.0 + 3.0) / 3.0
        np.testing.assert_allclose(result[0, 0, 0, 0], expected, rtol=1e-10)

    @pytest.mark.unit
    def test_block_average_fully_masked_bin_returns_zero(self, fitter):
        """A bin where all pixels are masked returns 0.0."""
        array = np.ones((2, 2, 1, 1), dtype=np.float64) * 5.0
        mask = np.zeros((2, 2, 1), dtype=bool)  # all masked

        result = fitter._block_average_array(array, (1, 1, 1), mask=mask)
        assert result[0, 0, 0, 0] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_block_average_no_mask_equivalent_to_mean(self, fitter):
        """Without a mask, block average equals the plain spatial mean of each bin."""
        rng = np.random.default_rng(42)
        array = rng.random((4, 4, 1, 3))
        result = fitter._block_average_array(array, (2, 2, 1))

        # Manually compute expected 2×2 block means
        for bi in range(2):
            for bj in range(2):
                block = array[bi * 2 : (bi + 1) * 2, bj * 2 : (bj + 1) * 2, 0, :]
                expected = block.mean(axis=(0, 1))
                np.testing.assert_allclose(
                    result[bi, bj, 0, :], expected, rtol=1e-10
                )


# ---------------------------------------------------------------------------
# TestIDEALFitterFit
# ---------------------------------------------------------------------------


class TestIDEALFitterFit:
    """Integration tests for the IDEALFitter.fit() workflow."""

    @pytest.mark.integration
    def test_fit_returns_self(self, fitter, b_values, image_4d):
        """fit() returns self to enable method chaining."""
        result = fitter.fit(b_values, image_4d)
        assert result is fitter

    @pytest.mark.integration
    def test_fit_sets_image_shape(self, fitter, b_values, image_4d):
        """fit() stores the full 4-D image shape as image_shape."""
        fitter.fit(b_values, image_4d)
        assert fitter.image_shape == image_4d.shape

    @pytest.mark.integration
    def test_fit_sets_n_measurements(self, fitter, b_values, image_4d):
        """fit() sets n_measurements to the number of b-values."""
        fitter.fit(b_values, image_4d)
        assert fitter.n_measurements == N_B

    @pytest.mark.integration
    def test_fit_populates_fitted_params_keys(self, fitter, b_values, image_4d):
        """fit() populates fitted_params_ with keys matching the model's param_names."""
        fitter.fit(b_values, image_4d)
        assert set(fitter.fitted_params_.keys()) == {"S0", "D"}

    @pytest.mark.integration
    def test_fit_step_params_length_matches_n_steps(
        self, fitter, b_values, image_4d, dim_steps
    ):
        """fit() appends one param map to step_params per IDEAL step."""
        fitter.fit(b_values, image_4d)
        n_steps = dim_steps.shape[0]  # n_steps is now first dimension
        assert len(fitter.step_params) == n_steps

    @pytest.mark.integration
    def test_fit_step_params_final_shape(self, fitter, b_values, image_4d):
        """The last entry in step_params has shape matching full-res spatial dims + n_params."""
        fitter.fit(b_values, image_4d)
        last_map = fitter.step_params[-1]
        # shape should be (4, 4, 1, n_params=2)
        assert last_map.shape == (4, 4, 1, 2)

    @pytest.mark.integration
    def test_fit_with_segmentation_restricts_pixels(self, fitter, b_values, image_4d):
        """Passing a partial segmentation fits only the masked voxels."""
        seg = np.zeros((4, 4, 1), dtype=int)
        seg[:2, :2, 0] = 1  # top-left 4 voxels
        fitter.fit(b_values, image_4d, segmentation=seg)
        # pixel_indices of the final step correspond to the masked region
        assert fitter.pixel_indices is not None
        assert len(fitter.pixel_indices) == 4

    @pytest.mark.integration
    def test_second_fit_resets_step_params(self, fitter, b_values, image_4d):
        """Calling fit() a second time resets step_params instead of appending."""
        fitter.fit(b_values, image_4d)
        n_steps_first = len(fitter.step_params)
        fitter.fit(b_values, image_4d)
        assert len(fitter.step_params) == n_steps_first, (
            "step_params should be reset on re-fit, not accumulated"
        )

    @pytest.mark.integration
    def test_fit_3d_image_expanded_to_4d(self, fitter, b_values):
        """fit() accepts a 3-D image (x, y, N_B) and expands it internally to 4-D."""
        image_3d = make_monoexp_image(n_x=4, n_y=4, n_z=1)[..., 0, :]  # shape (4,4,N_B)
        # Should not raise; image_shape stored should be the 4-D expanded shape
        fitter.fit(b_values, image_3d)
        assert fitter.image_shape[2] == 1, "Slice dimension should be 1 after expansion"

    @pytest.mark.integration
    def test_fit_with_block_average_downsampling(self, solver, dim_steps, step_tol, b_values, image_4d):
        """fit() completes successfully when downsampling_method='block_average'."""
        f = IDEALFitter(
            solver=solver,
            dim_steps=dim_steps,
            step_tol=step_tol,
            downsampling_method="block_average",
        )
        f.fit(b_values, image_4d)
        assert set(f.fitted_params_.keys()) == {"S0", "D"}, (
            "fitted_params_ should contain all model parameter keys"
        )

    @pytest.mark.integration
    def test_fit_with_cv2_area_downsampling(self, solver, dim_steps, step_tol, b_values, image_4d):
        """fit() completes successfully when downsampling_method='area'."""
        f = IDEALFitter(
            solver=solver,
            dim_steps=dim_steps,
            step_tol=step_tol,
            downsampling_method="area",
        )
        f.fit(b_values, image_4d)
        assert set(f.fitted_params_.keys()) == {"S0", "D"}


# ---------------------------------------------------------------------------
# TestIDEALFitterPredict
# ---------------------------------------------------------------------------


class TestIDEALFitterPredict:
    """Tests for the predict() method."""

    @pytest.mark.unit
    def test_predict_basic(self, fitter, b_values):
        """predict() returns a spatial volume matching image_shape and model.forward output."""
        xdata = b_values

        # configure a small image shape and pixel indices for prediction mapping
        fitter.image_shape = (2, 2, 1, xdata.size)
        fitter.n_measurements = xdata.size
        fitter.pixel_indices = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]

        # Create synthetic fitted parameters for each model parameter
        param_names = fitter.solver.model.param_names
        n_pixels = len(fitter.pixel_indices)
        param_arrays = [
            np.arange(1, n_pixels + 1) + 10 * i for i in range(len(param_names))
        ]
        fitter.fitted_params_ = {
            name: vals for name, vals in zip(param_names, param_arrays)
        }

        preds = fitter.predict(xdata)

        assert preds.shape == (2, 2, 1, xdata.size)

        # verify predictions for first and last pixel by calling model.forward directly
        params_pixel0 = [arr[0] for arr in param_arrays]
        expected0 = fitter.solver.model.forward(xdata, *params_pixel0)
        np.testing.assert_allclose(preds[0, 0, 0, :], expected0)

        params_pixel_last = [arr[-1] for arr in param_arrays]
        expected_last = fitter.solver.model.forward(xdata, *params_pixel_last)
        np.testing.assert_allclose(preds[1, 1, 0, :], expected_last)

    @pytest.mark.unit
    def test_predict_requires_fit(self, fitter, b_values):
        """predict() raises RuntimeError when fitter has not been fitted."""
        # ensure fitted_params_ is empty
        fitter.fitted_params_ = {}
        with pytest.raises(RuntimeError):
            fitter.predict(b_values)
