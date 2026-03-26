"""Tests for pyneapple.utility.validation — shape, segmentation, and parameter utilities."""

import numpy as np
import pytest

from pyneapple.utility import validation


# ---------------------------------------------------------------------------
# validate_data_shapes
# ---------------------------------------------------------------------------


class TestValidateDataShapes:
    """Tests for validate_data_shapes()."""

    def test_valid_1d_matching_lengths(self):
        """No error when 1D xdata and 1D ydata have equal length."""
        validation.validate_data_shapes(np.zeros(5), np.zeros(5))

    def test_valid_2d_multi_voxel(self):
        """No error for (n_voxels, n_b) ydata matching xdata length."""
        validation.validate_data_shapes(np.zeros(8), np.zeros((10, 8)))

    def test_valid_3d_image(self):
        """No error for (H, W, n_b) ydata where last dim matches xdata."""
        validation.validate_data_shapes(np.zeros(8), np.zeros((4, 5, 8)))

    def test_xdata_2d_raises(self):
        """ValueError when xdata is not 1D."""
        with pytest.raises(ValueError, match="xdata must be a 1D array"):
            validation.validate_data_shapes(np.zeros((2, 5)), np.zeros(5))

    def test_1d_length_mismatch_raises(self):
        """ValueError when 1D ydata length does not match xdata."""
        with pytest.raises(ValueError, match="does not match xdata length"):
            validation.validate_data_shapes(np.zeros(5), np.zeros(4))

    def test_2d_last_dim_mismatch_raises(self):
        """ValueError when last dimension of 2D ydata does not match xdata."""
        with pytest.raises(ValueError, match="does not match xdata length"):
            validation.validate_data_shapes(np.zeros(8), np.zeros((10, 7)))

    def test_3d_last_dim_mismatch_raises(self):
        """ValueError when last dimension of 3D ydata does not match xdata."""
        with pytest.raises(ValueError, match="does not match xdata length"):
            validation.validate_data_shapes(np.zeros(8), np.zeros((4, 5, 9)))


# ---------------------------------------------------------------------------
# validate_segmentation
# ---------------------------------------------------------------------------


class TestValidateSegmentation:
    """Tests for validate_segmentation()."""

    def test_valid_2d_segmentation(self):
        """Valid 2D segmentation matching (H, W, n_b) image returns unchanged array."""
        seg = np.zeros((4, 5), dtype=int)
        result = validation.validate_segmentation(seg, image_shape=(4, 5, 8))
        assert result.shape == (4, 5)

    def test_singleton_channel_squeezed(self):
        """Segmentation with trailing singleton dim is squeezed (loguru warning, not Python Warning)."""
        seg = np.zeros((4, 5, 1), dtype=int)
        result = validation.validate_segmentation(seg, image_shape=(4, 5, 8))
        assert result.shape == (4, 5)

    def test_shape_mismatch_raises(self):
        """ValueError when segmentation spatial shape does not match image."""
        seg = np.zeros((4, 6), dtype=int)
        with pytest.raises(ValueError, match="does not match expected image shape"):
            validation.validate_segmentation(seg, image_shape=(4, 5, 8))

    def test_wrong_ndim_non_singleton_raises(self):
        """ValueError for 3D segmentation with non-singleton last axis."""
        seg = np.zeros((4, 5, 2), dtype=int)
        with pytest.raises(ValueError, match="one less dimension"):
            validation.validate_segmentation(seg, image_shape=(4, 5, 8))

    def test_1d_segmentation_raises(self):
        """ValueError when segmentation is 1D for a 3D image."""
        seg = np.zeros(20, dtype=int)
        with pytest.raises(ValueError):
            validation.validate_segmentation(seg, image_shape=(4, 5, 8))


# ---------------------------------------------------------------------------
# validate_parameter_names
# ---------------------------------------------------------------------------


class TestValidateParameterNames:
    """Tests for validate_parameter_names()."""

    def test_valid_exact_match(self):
        """No error when all required parameter names are present."""
        validation.validate_parameter_names(
            {"S0": 1000.0, "D": 0.001}, param_names=["S0", "D"]
        )

    def test_missing_parameter_raises(self):
        """ValueError raised when a required parameter is missing."""
        with pytest.raises(ValueError, match="Missing"):
            validation.validate_parameter_names({"S0": 1000.0}, param_names=["S0", "D"])

    def test_all_missing_raises(self):
        """ValueError raised when all parameters are missing."""
        with pytest.raises(ValueError, match="Missing"):
            validation.validate_parameter_names({}, param_names=["S0", "D"])

    def test_extra_parameter_does_not_raise(self):
        """No error for extra parameters (they are ignored with a warning)."""
        validation.validate_parameter_names(
            {"S0": 1000.0, "D": 0.001, "extra": 9.9}, param_names=["S0", "D"]
        )


# ---------------------------------------------------------------------------
# transform_p0
# ---------------------------------------------------------------------------


class TestTransformP0:
    """Tests for transform_p0()."""

    PARAM_NAMES = ["S0", "D"]
    P0_DICT = {"S0": 1000.0, "D": 0.001}

    def test_scalar_no_pixels_returns_1d(self):
        """Without n_pixels, returns 1D array of shape (n_params,)."""
        result = validation.transform_p0(self.P0_DICT, self.PARAM_NAMES)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [1000.0, 0.001])

    def test_scalar_with_pixels_returns_2d(self):
        """With n_pixels=5, returns array of shape (n_params, n_pixels)."""
        result = validation.transform_p0(self.P0_DICT, self.PARAM_NAMES, n_pixels=5)
        assert result.shape == (2, 5), f"Expected (2, 5), got {result.shape}"

    def test_values_broadcast_correctly(self):
        """Each column in result equals the scalar p0 values."""
        result = validation.transform_p0(self.P0_DICT, self.PARAM_NAMES, n_pixels=3)
        np.testing.assert_array_equal(result[:, 0], result[:, 1])
        np.testing.assert_array_equal(result[:, 0], result[:, 2])
        assert result[0, 0] == pytest.approx(1000.0)
        assert result[1, 0] == pytest.approx(0.001)

    def test_order_follows_param_names(self):
        """Values appear in param_names order, not dict insertion order."""
        p0 = {"D": 0.001, "S0": 1000.0}  # reversed order
        result = validation.transform_p0(p0, ["S0", "D"])
        assert result[0] == pytest.approx(1000.0), "S0 should be first"
        assert result[1] == pytest.approx(0.001), "D should be second"

    def test_int_values_accepted(self):
        """Integer values in p0 dict are accepted (not just float)."""
        p0 = {"S0": 1000, "D": 0.001}  # S0 is int
        result = validation.transform_p0(p0, self.PARAM_NAMES)
        assert result[0] == pytest.approx(1000.0)

    def test_missing_parameter_raises(self):
        """ValueError raised when a required parameter is absent from p0."""
        with pytest.raises(ValueError, match="Missing"):
            validation.transform_p0({"S0": 1000.0}, self.PARAM_NAMES)

    def test_wrong_value_type_raises(self):
        """ValueError raised when p0 values are not scalar int/float."""
        with pytest.raises(ValueError):
            validation.transform_p0({"S0": [1000.0], "D": [0.001]}, self.PARAM_NAMES)


# ---------------------------------------------------------------------------
# transform_bounds
# ---------------------------------------------------------------------------


class TestTransformBounds:
    """Tests for transform_bounds()."""

    PARAM_NAMES = ["S0", "D"]
    BOUNDS_DICT = {"S0": (0.0, 5000.0), "D": (0.0, 0.05)}

    def test_no_pixels_returns_1d(self):
        """Without n_pixels, returns two 1D arrays of shape (n_params,)."""
        lower, upper = validation.transform_bounds(self.BOUNDS_DICT, self.PARAM_NAMES)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        np.testing.assert_array_equal(lower, [0.0, 0.0])
        np.testing.assert_array_equal(upper, [5000.0, 0.05])

    def test_with_pixels_returns_correct_shape(self):
        """With n_pixels=4, returns arrays of shape (n_params, n_pixels)."""
        lower, upper = validation.transform_bounds(
            self.BOUNDS_DICT, self.PARAM_NAMES, n_pixels=4
        )
        assert lower.shape == (2, 4), f"Expected (2, 4), got {lower.shape}"
        assert upper.shape == (2, 4), f"Expected (2, 4), got {upper.shape}"

    def test_values_broadcast_correctly(self):
        """All pixel columns contain the same bound values."""
        lower, upper = validation.transform_bounds(
            self.BOUNDS_DICT, self.PARAM_NAMES, n_pixels=3
        )
        np.testing.assert_array_equal(lower[:, 0], lower[:, 1])
        assert lower[0, 0] == pytest.approx(0.0)  # S0 lower
        assert upper[0, 0] == pytest.approx(5000.0)  # S0 upper
        assert lower[1, 0] == pytest.approx(0.0)  # D lower
        assert upper[1, 0] == pytest.approx(0.05)  # D upper

    def test_order_follows_param_names(self):
        """Bounds appear in param_names order, not dict insertion order."""
        bounds = {"D": (0.0, 0.05), "S0": (0.0, 5000.0)}  # reversed order
        lower, upper = validation.transform_bounds(bounds, ["S0", "D"])
        assert lower[0] == pytest.approx(0.0)
        assert upper[0] == pytest.approx(5000.0)  # S0 upper

    def test_missing_parameter_raises(self):
        """ValueError raised when a required parameter is absent from bounds."""
        with pytest.raises(ValueError, match="Missing"):
            validation.transform_bounds({"S0": (0.0, 5000.0)}, self.PARAM_NAMES)

    def test_wrong_value_type_raises(self):
        """ValueError raised when bound values are not floats."""
        with pytest.raises(ValueError):
            validation.transform_bounds(
                {"S0": ([0.0], [5000.0]), "D": ([0.0], [0.05])}, self.PARAM_NAMES
            )


# ---------------------------------------------------------------------------
# transform_p0_spatial
# ---------------------------------------------------------------------------


class TestTransformP0Spatial:
    """Tests for transform_p0_spatial()."""

    IMAGE_SHAPE = (3, 3)
    PARAM_NAMES = ["S0", "D"]

    @pytest.fixture
    def spatial_p0(self):
        """Spatial p0 arrays shaped (H, W) for each parameter."""
        return {
            "S0": np.full(self.IMAGE_SHAPE, 1000.0),
            "D": np.full(self.IMAGE_SHAPE, 0.001),
        }

    def test_output_shape_all_pixels(self, spatial_p0):
        """Returns (n_pixels, n_params) when pixel_indices is None."""
        result = validation.transform_p0_spatial(
            spatial_p0, self.PARAM_NAMES, self.IMAGE_SHAPE
        )
        n_pixels = self.IMAGE_SHAPE[0] * self.IMAGE_SHAPE[1]
        assert result.shape == (n_pixels, len(self.PARAM_NAMES))

    def test_output_shape_subset_pixels(self, spatial_p0):
        """Returns (len(pixel_indices), n_params) when pixel_indices is provided."""
        pixel_indices = [(0, 0), (1, 1), (2, 2)]
        result = validation.transform_p0_spatial(
            spatial_p0, self.PARAM_NAMES, self.IMAGE_SHAPE, pixel_indices=pixel_indices
        )
        assert result.shape == (3, len(self.PARAM_NAMES))

    def test_values_match_spatial_arrays(self, spatial_p0):
        """Output values match the corresponding entries in the input arrays."""
        spatial_p0["S0"][0, 0] = 500.0
        pixel_indices = [(0, 0)]
        result = validation.transform_p0_spatial(
            spatial_p0, self.PARAM_NAMES, self.IMAGE_SHAPE, pixel_indices=pixel_indices
        )
        assert result[0, 0] == pytest.approx(500.0), "S0 at (0,0) should be 500"

    def test_wrong_shape_raises(self, spatial_p0):
        """ValueError raised when a parameter array has the wrong shape."""
        spatial_p0["S0"] = np.zeros((2, 4))  # wrong shape
        with pytest.raises(ValueError, match="must have shape"):
            validation.transform_p0_spatial(
                spatial_p0, self.PARAM_NAMES, self.IMAGE_SHAPE
            )

    def test_non_array_value_raises(self, spatial_p0):
        """ValueError raised when a parameter value is not a numpy array."""
        spatial_p0["S0"] = 1000.0  # scalar instead of array
        with pytest.raises(ValueError, match="must be a numpy array"):
            validation.transform_p0_spatial(
                spatial_p0, self.PARAM_NAMES, self.IMAGE_SHAPE
            )


# ---------------------------------------------------------------------------
# transform_bounds_spatial
# ---------------------------------------------------------------------------


class TestTransformBoundsSpatial:
    """Tests for transform_bounds_spatial()."""

    IMAGE_SHAPE = (3, 3)
    PARAM_NAMES = ["S0", "D"]

    @pytest.fixture
    def spatial_bounds(self):
        """Spatial bounds arrays shaped (H, W) for lower and upper per parameter."""
        return {
            "S0": (np.zeros(self.IMAGE_SHAPE), np.full(self.IMAGE_SHAPE, 5000.0)),
            "D": (np.zeros(self.IMAGE_SHAPE), np.full(self.IMAGE_SHAPE, 0.05)),
        }

    def test_output_shape_all_pixels(self, spatial_bounds):
        """Returns two (n_pixels, n_params) arrays when pixel_indices is None."""
        lower, upper = validation.transform_bounds_spatial(
            spatial_bounds, self.PARAM_NAMES, self.IMAGE_SHAPE
        )
        n_pixels = self.IMAGE_SHAPE[0] * self.IMAGE_SHAPE[1]
        assert lower.shape == (n_pixels, len(self.PARAM_NAMES))
        assert upper.shape == (n_pixels, len(self.PARAM_NAMES))

    def test_values_match_spatial_arrays(self, spatial_bounds):
        """Output bound values match the corresponding entries in the input arrays."""
        spatial_bounds["S0"][1][0, 0] = 999.0  # set upper bound at (0, 0)
        pixel_indices = [(0, 0)]
        lower, upper = validation.transform_bounds_spatial(
            spatial_bounds,
            self.PARAM_NAMES,
            self.IMAGE_SHAPE,
            pixel_indices=pixel_indices,
        )
        assert upper[0, 0] == pytest.approx(999.0), "S0 upper at (0,0) should be 999"

    def test_non_array_raises(self, spatial_bounds):
        """ValueError raised when bound arrays are not numpy arrays."""
        spatial_bounds["S0"] = (0.0, 5000.0)  # scalars instead of arrays
        with pytest.raises(ValueError, match="must be numpy arrays"):
            validation.transform_bounds_spatial(
                spatial_bounds, self.PARAM_NAMES, self.IMAGE_SHAPE
            )

    def test_wrong_shape_raises(self, spatial_bounds):
        """ValueError raised when a bound array has the wrong spatial shape."""
        spatial_bounds["S0"] = (np.zeros((2, 4)), np.ones((2, 4)))
        with pytest.raises(ValueError, match="must have shape"):
            validation.transform_bounds_spatial(
                spatial_bounds, self.PARAM_NAMES, self.IMAGE_SHAPE
            )

    def test_missing_parameter_raises(self, spatial_bounds):
        """ValueError raised when a required parameter is absent from bounds."""
        with pytest.raises(ValueError, match="Missing"):
            validation.transform_bounds_spatial(
                {"S0": spatial_bounds["S0"]}, self.PARAM_NAMES, self.IMAGE_SHAPE
            )


# ---------------------------------------------------------------------------
# validate_fixed_params
# ---------------------------------------------------------------------------


class TestValidateFixedParams:
    """Tests for validate_fixed_params()."""

    ALL_NAMES = ["S0", "D", "T1"]

    def test_empty_dict_passes(self):
        """No error for an empty fixed_params dict."""
        validation.validate_fixed_params({}, self.ALL_NAMES)

    def test_none_passes(self):
        """No error for None (treated as empty)."""
        validation.validate_fixed_params(None, self.ALL_NAMES)

    def test_valid_subset_passes(self):
        """No error when all keys are valid parameter names."""
        validation.validate_fixed_params({"D": 0.001}, self.ALL_NAMES)

    def test_unknown_key_raises(self):
        """ValueError when a key is not in all_param_names."""
        with pytest.raises(ValueError, match="Unknown fixed parameter"):
            validation.validate_fixed_params({"FAKE": 1.0}, self.ALL_NAMES)

    def test_fix_all_raises(self):
        """ValueError when fixing every parameter."""
        with pytest.raises(ValueError, match="Cannot fix all"):
            validation.validate_fixed_params(
                {"S0": 1.0, "D": 0.001, "T1": 1200.0}, self.ALL_NAMES
            )


# ---------------------------------------------------------------------------
# validate_fixed_param_maps
# ---------------------------------------------------------------------------


class TestValidateFixedParamMaps:
    """Tests for validate_fixed_param_maps()."""

    ALL_NAMES = ["S0", "D"]
    SHAPE = (4, 4, 1)

    def test_valid_map_passes(self):
        """No error for correctly shaped arrays with valid names."""
        maps = {"S0": np.ones(self.SHAPE)}
        validation.validate_fixed_param_maps(maps, self.SHAPE, self.ALL_NAMES)

    def test_wrong_shape_raises(self):
        """ValueError when array shape does not match spatial_shape."""
        maps = {"S0": np.ones((2, 2, 1))}
        with pytest.raises(ValueError, match="has shape"):
            validation.validate_fixed_param_maps(maps, self.SHAPE, self.ALL_NAMES)

    def test_unknown_name_raises(self):
        """ValueError when a key is not a valid param name."""
        maps = {"BOGUS": np.ones(self.SHAPE)}
        with pytest.raises(ValueError, match="Unknown fixed parameter"):
            validation.validate_fixed_param_maps(maps, self.SHAPE, self.ALL_NAMES)
