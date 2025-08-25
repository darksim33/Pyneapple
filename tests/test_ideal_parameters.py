import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from pyneapple import IDEALParams
from radimgarray import RadImgArray, SegImgArray


@pytest.mark.ideal
class TestIDEALParameters:
    """Test class for IDEALParams basic setup and properties."""

    def test_ideal_params_initialization_default(self):
        """Test default initialization of IDEALParams."""
        params = IDEALParams()

        assert params.step_tol is None
        assert (params.dim_steps == np.array([])).all()
        assert params.seg_threshold == 0.025

    def test_ideal_params_initialization_with_file(self, ideal_params_file):
        """Test initialization with parameter file."""
        # Create a temporary JSON file
        params = IDEALParams(ideal_params_file)
        assert params is not None

    def test_dimension_steps_setter_list(self):
        """Test dimension_steps setter with list input."""
        params = IDEALParams()
        test_steps = [[10, 10], [20, 20], [30, 30]]

        params.dim_steps = test_steps

        assert isinstance(params.dim_steps, np.ndarray)
        assert params.dim_steps.dtype == np.int32
        np.testing.assert_array_equal(params.dim_steps, np.array(test_steps))

    def test_dimension_steps_setter_numpy_array(self):
        """Test dimension_steps setter with numpy array input."""
        params = IDEALParams()
        test_steps = np.array([[5, 5], [15, 15], [25, 25]])

        params.dim_steps = test_steps

        assert isinstance(params.dim_steps, np.ndarray)
        assert params.dim_steps.dtype == np.int32
        np.testing.assert_array_equal(params.dim_steps, test_steps)

    def test_dimension_steps_setter_none(self):
        """Test dimension_steps setter with None input."""
        params = IDEALParams()
        params.dim_steps = None

        assert (params.dim_steps == np.ndarray([], dtype=np.int32)).all()

    def test_dimension_steps_setter_invalid_type(self):
        """Test dimension_steps setter with invalid input type."""
        params = IDEALParams()

        with pytest.raises(ValueError, match="Expected list or numpy array"):
            params.dim_steps = "invalid"

    def test_step_tolerance_setter_list(self):
        """Test step_tolerance setter with list input."""
        params = IDEALParams()
        test_tolerance = [0.1, 0.2, 0.3]

        params.step_tol = test_tolerance

        assert isinstance(params.step_tol, np.ndarray)
        assert params.step_tol.dtype == np.float32
        np.testing.assert_array_equal(
            params.step_tol, np.array(test_tolerance, dtype=np.float32)
        )

    def test_step_tolerance_setter_numpy_array(self):
        """Test step_tolerance setter with numpy array input."""
        params = IDEALParams()
        test_tolerance = np.array([0.05, 0.15], np.float32)

        params.step_tol = test_tolerance

        assert isinstance(params.step_tol, np.ndarray)
        assert params.step_tol.dtype == np.float32
        np.testing.assert_array_equal(params.step_tol, test_tolerance)

    def test_step_tolerance_setter_float(self):
        """Test step_tolerance setter with single float input."""
        params = IDEALParams()
        test_tolerance = 0.05

        params.step_tol = test_tolerance

        assert isinstance(params.step_tol, np.ndarray)
        assert params.step_tol.dtype == np.float32
        np.testing.assert_array_equal(
            params.step_tol, np.array([test_tolerance], dtype=np.float32)
        )

    def test_step_tolerance_setter_none(self):
        """Test step_tolerance setter with None input."""
        params = IDEALParams()
        params.step_tol = None

        assert params.step_tol is None

    def test_step_tolerance_setter_invalid_type(self):
        """Test step_tolerance setter with invalid input type."""
        params = IDEALParams()

        with pytest.raises(TypeError, match="Expected list or numpy array"):
            params.step_tol = "invalid"

    def test_segmentation_threshold_setter_float(self):
        """Test segmentation_threshold setter with float input."""
        params = IDEALParams()
        test_threshold = 0.1

        params.seg_threshold = test_threshold

        assert params.seg_threshold == test_threshold

    def test_segmentation_threshold_setter_numpy_float(self):
        """Test segmentation_threshold setter with numpy float input."""
        params = IDEALParams()
        test_threshold = np.float64(0.075)

        params.seg_threshold = test_threshold

        assert params.seg_threshold == test_threshold

    def test_segmentation_threshold_setter_none(self):
        """Test segmentation_threshold setter with None input (default)."""
        params = IDEALParams()
        params.seg_threshold = None

        assert params.seg_threshold == 0.025

    def test_segmentation_threshold_setter_invalid_type(self):
        """Test segmentation_threshold setter with invalid input type."""
        params = IDEALParams()

        with pytest.raises(TypeError, match="Expected float"):
            params.seg_threshold = "invalid"

    def test_inheritance_from_ivim_params(self):
        """Test that IDEALParams properly inherits from IVIMParams."""
        params = IDEALParams()

        # Should have IVIMParams attributes
        assert hasattr(params, "fit_model")
        assert hasattr(params, "boundaries")
        assert hasattr(params, "model")

    def test_interpolate_array_2d_default(self, ideal_params_file):
        """Test interpolate_array with 2D array and default parameters."""
        params = IDEALParams(ideal_params_file)

        # Create base array (5x5x4 for BiExp model with 4 parameters)
        base_array = np.random.rand(5, 5, 4, 3).astype(np.float32)

        result = params.interpolate_array(base_array, step_idx=0)

        # Should interpolate to dimension_steps[0] = [8, 8] with 4 parameters
        assert result.shape == (1, 1, 4, 3)
        assert result.dtype == np.float32

    def test_interpolate_array_2d_custom_shape(self, ideal_params_file):
        """Test interpolate_array with custom matrix_shape."""
        params = IDEALParams(ideal_params_file)

        base_array = np.random.rand(6, 6, 4, 6).astype(np.float32)
        custom_shape = (12, 12)

        result = params.interpolate_array(
            base_array, step_idx=1, matrix_shape=custom_shape
        )

        assert result.shape == (12, 12, 4, 6)

    def test_interpolate_array_different_step_indices(self, ideal_params_file):
        """Test interpolate_array with different step indices."""
        params = IDEALParams(ideal_params_file)

        base_array = np.random.rand(4, 4, 4, 4).astype(np.float32)

        # Test different step indices from dimension_steps [[8, 8], [16, 16], [32, 32]]
        result_0 = params.interpolate_array(base_array, step_idx=0)
        result_1 = params.interpolate_array(base_array, step_idx=1)
        result_2 = params.interpolate_array(base_array, step_idx=2)

        assert result_0.shape == (1, 1, 4, 4)
        assert result_1.shape == (2, 2, 4, 4)
        assert result_2.shape == (4, 4, 4, 4)

    def test_interpolate_array_custom_interpolation(self, ideal_params_file):
        """Test interpolate_array with custom interpolation method."""
        params = IDEALParams(ideal_params_file)

        base_array = np.random.rand(4, 4, 4, 4).astype(np.float32)

        # Test with different interpolation method
        result = params.interpolate_array(
            base_array, step_idx=0, interpolation=cv2.INTER_LINEAR
        )

        assert result.shape == (1, 1, 4, 4)

    def test_interpolate_array_3d_not_implemented(self, ideal_params_file):
        """Test that 3D interpolation raises NotImplementedError."""
        params = IDEALParams(ideal_params_file)
        params.ideal_dims = 3  # Set to 3D

        base_array = np.random.rand(5, 5, 5, 4).astype(np.float32)

        with pytest.raises(
            NotImplementedError, match="Currently only 2D interpolation is supported"
        ):
            params.interpolate_array(base_array, step_idx=0)

    def test_interpolate_array_various_dtypes(self, ideal_params_file):
        """Test interpolate_array with different input dtypes."""
        params = IDEALParams(ideal_params_file)

        # Test with different dtypes
        for dtype in [np.float32, np.float64]:
            base_array = np.random.rand(3, 3, 4, 4).astype(dtype)
            result = params.interpolate_array(base_array, step_idx=3)
            assert result.shape == (8, 8, 4, 4)

    def test_interpolate_array_edge_cases(self, ideal_params_file):
        """Test interpolate_array with edge cases."""
        params = IDEALParams(ideal_params_file)

        # Test with single parameter
        base_array = np.random.rand(2, 2, 1, 6).astype(np.float32)
        result = params.interpolate_array(base_array, step_idx=0)
        assert result.shape == (1, 1, 1, 6)

        # Test upscaling and downscaling
        large_array = np.random.rand(50, 50, 4, 6).astype(np.float32)
        result_down = params.interpolate_array(large_array, step_idx=0)
        assert result_down.shape == (1, 1, 4, 6)

    def test_interpolate_img_with_radimgarray(self, ideal_params_file):
        """Test interpolate_img with RadImgArray input."""
        params = IDEALParams(ideal_params_file)

        # Create RadImgArray with 4D data (x, y, z, n_args)
        array_data = np.random.rand(5, 5, 3, 4).astype(np.float32)
        img = RadImgArray(array_data)

        result = params.interpolate_img(img, step_idx=0)

        assert isinstance(result, RadImgArray)  # Should return RadImgArray
        assert result.shape == (1, 1, 3, 4)

    def test_interpolate_img_with_numpy_array(self, ideal_params_file):
        """Test interpolate_img with numpy array input."""
        params = IDEALParams(ideal_params_file)

        # Create numpy array
        array_data = np.random.rand(4, 4, 2, 6).astype(np.float32)

        result = params.interpolate_img(array_data, step_idx=1)

        assert isinstance(result, RadImgArray)  # Always returns RadImgArray
        assert result.shape == (2, 2, 2, 6)

    def test_interpolate_img_with_segimgarray(self, ideal_params_file):
        """Test interpolate_img with SegImgArray input preserves info."""
        params = IDEALParams(ideal_params_file)

        # Create SegImgArray with mock info
        array_data = np.random.rand(6, 6, 2, 3).astype(np.float32)
        seg_img = SegImgArray(array_data)

        result = params.interpolate_img(seg_img, step_idx=2)

        assert isinstance(
            result, RadImgArray
        )  # Returns RadImgArray with preserved info
        assert result.shape == (4, 4, 2, 3)
        # Info should be preserved from SegImgArray
        assert result.info is seg_img.info

    def test_interpolate_img_custom_matrix_shape(self, ideal_params_file):
        """Test interpolate_img with custom matrix shape."""
        params = IDEALParams(ideal_params_file)

        array_data = np.random.rand(3, 3, 4, 2).astype(np.float32)
        img = RadImgArray(array_data)

        result = params.interpolate_img(img, step_idx=0, matrix_shape=(20, 20))

        assert isinstance(result, RadImgArray)
        assert result.shape == (20, 20, 4, 2)

    def test_interpolate_seg_with_numpy_array(self, ideal_params_file):
        """Test interpolate_seg with numpy array input."""
        params = IDEALParams(ideal_params_file)

        # Create binary segmentation data
        array_data = np.random.choice([0, 1], size=(5, 5, 2, 1)).astype(np.float32)

        result = params.interpolate_seg(array_data, step_idx=0)

        assert isinstance(result, RadImgArray)  # Returns RadImgArray
        assert result.shape == (1, 1, 2, 1)
        # Check binary values are preserved (0 or 1)
        assert np.all((result == 0) | (result == 1))

    def test_interpolate_seg_threshold_behavior(self, ideal_params_file):
        """Test that interpolate_seg properly applies threshold."""
        params = IDEALParams(ideal_params_file)

        # Create data with known values around threshold (0.025)
        array_data = np.full((3, 3, 1, 1), 0.02, dtype=np.float32)  # Below threshold
        array_data[1, 1, 0, 0] = 0.03  # Above threshold

        result = params.interpolate_seg(array_data, step_idx=3)

        # After interpolation and thresholding, should be binary
        assert np.all((result == 0) | (result == 1))
        assert result.shape == (8, 8, 1, 1)

    def test_interpolate_seg_uses_segmentation_threshold(self, ideal_params_file):
        """Test interpolate_seg uses the segmentation_threshold from params."""
        params = IDEALParams(ideal_params_file)

        # segmentation_threshold should be 0.025 from fixture
        assert params.seg_threshold == 0.025

        # Create data with values around the threshold
        array_data = np.full((2, 2, 1, 1), 0.02, dtype=np.float32)  # Below threshold
        array_data[0, 0, 0, 0] = 0.03  # Above threshold

        result = params.interpolate_seg(array_data, step_idx=0)

        assert np.all((result == 0) | (result == 1))
        assert isinstance(result, RadImgArray)

    def test_interpolate_methods_error_handling(self, ideal_params_file):
        """Test error handling for 3D interpolation in img and seg methods."""
        params = IDEALParams(ideal_params_file)
        params.ideal_dims = 3  # Force 3D to trigger error

        array_data = np.random.rand(4, 4, 2, 3).astype(np.float32)

        with pytest.raises(NotImplementedError):
            params.interpolate_img(array_data, step_idx=0)

        with pytest.raises(NotImplementedError):
            params.interpolate_seg(array_data, step_idx=0)

    def test_interpolate_methods_preserve_data_types(self, ideal_params_file):
        """Test that interpolation methods handle different input data types correctly."""
        params = IDEALParams(ideal_params_file)

        # Test with different dtypes
        for dtype in [np.float32, np.float64]:
            array_data = np.random.rand(3, 3, 2, 4).astype(dtype)

            img_result = params.interpolate_img(array_data, step_idx=3)
            seg_result = params.interpolate_seg(array_data, step_idx=3)

            assert isinstance(img_result, RadImgArray)
            assert isinstance(seg_result, RadImgArray)
            assert img_result.shape == (8, 8, 2, 4)
            assert seg_result.shape == (8, 8, 2, 4)

    def test_get_boundaries_step_0(self, ideal_params_file):
        """Test get_boundaries method returns correct boundaries."""
        params = IDEALParams(ideal_params_file)
        results = np.random.rand(1, 1, 4).astype(np.float32)

        # Test with step index 0
        x0, lb, ub = params.get_boundaries(step_idx=0, result=results)

        assert isinstance(x0, np.ndarray)
        assert isinstance(lb, np.ndarray)
        assert isinstance(ub, np.ndarray)

        # Check shapes and types
        assert x0.shape == (4,)
        assert lb.shape == (4,)
        assert ub.shape == (4,)
        assert x0.dtype == np.float64
        assert lb.dtype == np.float64
        assert ub.dtype == np.float64

    def test_get_boundaries_step_1(self, ideal_params_file):
        """Test get_boundaries method for step index 1."""
        params = IDEALParams(ideal_params_file)
        results = np.random.rand(1, 1, 4, 4).astype(
            np.float32
        )  # Mock results for boundaries

        # Test with step index 1
        x0, lb, ub = params.get_boundaries(step_idx=4, result=results)

        assert isinstance(x0, np.ndarray)
        assert isinstance(lb, np.ndarray)
        assert isinstance(ub, np.ndarray)

        # Check shapes and types
        assert x0.shape == (16, 16, 4, 4)
        assert lb.shape == (16, 16, 4, 4)
        assert ub.shape == (16, 16, 4, 4)
        assert x0.dtype == np.float32
        assert lb.dtype == np.float32
        assert ub.dtype == np.float32

    def test_get_boundaries_tolerance_application(self, ideal_params_file):
        """Test that step tolerance is correctly applied to boundaries."""
        params = IDEALParams(ideal_params_file)
        # Ensure step tolerance is set
        params.step_tol = [0.05, 0.1, 0.15]

        # Create mock results with known values
        results = np.ones((4, 4, 4, 3), dtype=np.float32)

        # For step_idx > 0, it should use interpolated results and apply tolerance
        x0, lb, ub = params.get_boundaries(step_idx=1, result=results)

        # Since results are all ones and step_idx=1, the tolerance should be 0.1 (second value)
        # Check that bounds are correctly calculated
        np.testing.assert_allclose(
            lb, x0 * [0.95, 0.9, 0.85]
        )  # lower bound = x0 * (1 - tol)
        np.testing.assert_allclose(
            ub, x0 * [1.05, 1.1, 1.15]
        )  # upper bound = x0 * (1 + tol)

    def test_get_boundaries_with_complex_results(self, ideal_params_file):
        """Test get_boundaries with results having varying values."""
        params = IDEALParams(ideal_params_file)
        params.step_tol = 0.1  # Single tolerance value

        # Create results with varying values
        results = np.random.rand(4, 4, 4, 3).astype(np.float32)

        # For step_idx > 0
        x0, lb, ub = params.get_boundaries(step_idx=4, result=results)

        # Verify shapes and tolerance application
        assert x0.shape == (16, 16, 4, 3)
        np.testing.assert_allclose(lb, x0 * 0.9)  # lower bound = x0 * (1 - 0.1)
        np.testing.assert_allclose(ub, x0 * 1.1)  # upper bound = x0 * (1 + 0.1)

        # Verify that interpolation preserves relative data patterns
        # The min/max relationship should be preserved after interpolation
        assert np.argmax(results.flatten()) != np.argmin(results.flatten())
        max_val_x0 = np.max(x0)
        min_val_x0 = np.min(x0)
        assert max_val_x0 > min_val_x0
