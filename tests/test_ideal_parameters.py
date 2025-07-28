import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from pyneapple import IDEALParams


@pytest.fixture
def ideal_params_file():
    """Create a temporary IDEAL parameter file."""
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
        f.write(
            b"""
            # Test IDEAL Parameter File
            [General]
            Class = "IDEALParams"
            fit_type = "single"
            max_iter = 100
            fit_tolerance = 1e-6
            n_pools = 4
            dims = 2
            step_tolerance = [0.01, 0.02, 0.03]
            dimension_steps = [[8, 8], [16, 16], [32, 32]]
            segmentation_threshold = 0.025
            
            [Model]
            model = "BiExp"
            fit_reduced = false
            fit_S0 = true

            [boundaries]
            
            [boundaries.D]
            "1" = [0.001, 0.0007, 0.05]
            "2" = [0.02, 0.003, 0.3]

            [boundaries.f]
            "1" = [85, 10, 500]
            "2" = [20, 1, 100]
            
            """
        )
        temp_file = f.name

    yield Path(temp_file)

    # Clean up
    if Path(temp_file).exists():
        Path(temp_file).unlink()


@pytest.mark.ideal
class TestIDEALParameters:
    """Test class for IDEALParams basic setup and properties."""

    def test_ideal_params_initialization_default(self):
        """Test default initialization of IDEALParams."""
        params = IDEALParams()

        assert params.step_tolerance is None
        assert params.dimension_steps is None
        assert params.segmentation_threshold == 0.025

    def test_ideal_params_initialization_with_file(self, ideal_params_file):
        """Test initialization with parameter file."""
        # Create a temporary JSON file
        params = IDEALParams(ideal_params_file)
        assert params is not None

    def test_dimension_steps_setter_list(self):
        """Test dimension_steps setter with list input."""
        params = IDEALParams()
        test_steps = [[10, 10], [20, 20], [30, 30]]

        params.dimension_steps = test_steps

        assert isinstance(params.dimension_steps, np.ndarray)
        assert params.dimension_steps.dtype == np.int32
        np.testing.assert_array_equal(params.dimension_steps, np.array(test_steps))

    def test_dimension_steps_setter_numpy_array(self):
        """Test dimension_steps setter with numpy array input."""
        params = IDEALParams()
        test_steps = np.array([[5, 5], [15, 15], [25, 25]])

        params.dimension_steps = test_steps

        assert isinstance(params.dimension_steps, np.ndarray)
        assert params.dimension_steps.dtype == np.int32
        np.testing.assert_array_equal(params.dimension_steps, test_steps)

    def test_dimension_steps_setter_none(self):
        """Test dimension_steps setter with None input."""
        params = IDEALParams()
        params.dimension_steps = None

        assert params.dimension_steps is None

    def test_dimension_steps_setter_invalid_type(self):
        """Test dimension_steps setter with invalid input type."""
        params = IDEALParams()

        with pytest.raises(ValueError, match="Expected list or numpy array"):
            params.dimension_steps = "invalid"

    def test_step_tolerance_setter_list(self):
        """Test step_tolerance setter with list input."""
        params = IDEALParams()
        test_tolerance = [0.1, 0.2, 0.3]

        params.step_tolerance = test_tolerance

        assert isinstance(params.step_tolerance, np.ndarray)
        assert params.step_tolerance.dtype == np.float32
        np.testing.assert_array_equal(
            params.step_tolerance, np.array(test_tolerance, dtype=np.float32)
        )

    def test_step_tolerance_setter_numpy_array(self):
        """Test step_tolerance setter with numpy array input."""
        params = IDEALParams()
        test_tolerance = np.array([0.05, 0.15], np.float32)

        params.step_tolerance = test_tolerance

        assert isinstance(params.step_tolerance, np.ndarray)
        assert params.step_tolerance.dtype == np.float32
        np.testing.assert_array_equal(params.step_tolerance, test_tolerance)

    def test_step_tolerance_setter_float(self):
        """Test step_tolerance setter with single float input."""
        params = IDEALParams()
        test_tolerance = 0.05

        params.step_tolerance = test_tolerance

        assert isinstance(params.step_tolerance, np.ndarray)
        assert params.step_tolerance.dtype == np.float32
        np.testing.assert_array_equal(
            params.step_tolerance, np.array([test_tolerance], dtype=np.float32)
        )

    def test_step_tolerance_setter_none(self):
        """Test step_tolerance setter with None input."""
        params = IDEALParams()
        params.step_tolerance = None

        assert params.step_tolerance is None

    def test_step_tolerance_setter_invalid_type(self):
        """Test step_tolerance setter with invalid input type."""
        params = IDEALParams()

        with pytest.raises(TypeError, match="Expected list or numpy array"):
            params.step_tolerance = "invalid"

    def test_segmentation_threshold_setter_float(self):
        """Test segmentation_threshold setter with float input."""
        params = IDEALParams()
        test_threshold = 0.1

        params.segmentation_threshold = test_threshold

        assert params.segmentation_threshold == test_threshold

    def test_segmentation_threshold_setter_numpy_float(self):
        """Test segmentation_threshold setter with numpy float input."""
        params = IDEALParams()
        test_threshold = np.float64(0.075)

        params.segmentation_threshold = test_threshold

        assert params.segmentation_threshold == test_threshold

    def test_segmentation_threshold_setter_none(self):
        """Test segmentation_threshold setter with None input (default)."""
        params = IDEALParams()
        params.segmentation_threshold = None

        assert params.segmentation_threshold == 0.025

    def test_segmentation_threshold_setter_invalid_type(self):
        """Test segmentation_threshold setter with invalid input type."""
        params = IDEALParams()

        with pytest.raises(TypeError, match="Expected float"):
            params.segmentation_threshold = "invalid"

    def test_inheritance_from_ivim_params(self):
        """Test that IDEALParams properly inherits from IVIMParams."""
        params = IDEALParams()

        # Should have IVIMParams attributes
        assert hasattr(params, "fit_model")
        assert hasattr(params, "boundaries")
        assert hasattr(params, "model")

    def test_interpolate_start_values_2d_default(self, ideal_params_file):
        """Test interpolate_start_values with 2D array and default parameters."""
        params = IDEALParams(ideal_params_file)

        # Create base array (5x5x4 for BiExp model with 4 parameters)
        base_array = np.random.rand(5, 5, 4, 3).astype(np.float32)

        result = params.interpolate_start_values(base_array, step_idx=0)

        # Should interpolate to dimension_steps[0] = [8, 8] with 4 parameters
        assert result.shape == (8, 8, 4, 3)
        assert result.dtype == np.float32

    def test_interpolate_start_values_2d_custom_shape(self, ideal_params_file):
        """Test interpolate_start_values with custom matrix_shape."""
        params = IDEALParams(ideal_params_file)

        base_array = np.random.rand(6, 6, 4, 6).astype(np.float32)
        custom_shape = (12, 12)

        result = params.interpolate_start_values(
            base_array, step_idx=1, matrix_shape=custom_shape
        )

        assert result.shape == (12, 12, 4, 6)

    def test_interpolate_start_values_different_step_indices(self, ideal_params_file):
        """Test interpolate_start_values with different step indices."""
        params = IDEALParams(ideal_params_file)

        base_array = np.random.rand(4, 4, 4, 4).astype(np.float32)

        # Test different step indices from dimension_steps [[8, 8], [16, 16], [32, 32]]
        result_0 = params.interpolate_start_values(base_array, step_idx=0)
        result_1 = params.interpolate_start_values(base_array, step_idx=1)
        result_2 = params.interpolate_start_values(base_array, step_idx=2)

        assert result_0.shape == (8, 8, 4, 4)
        assert result_1.shape == (16, 16, 4, 4)
        assert result_2.shape == (32, 32, 4, 4)

    def test_interpolate_start_values_custom_interpolation(self, ideal_params_file):
        """Test interpolate_start_values with custom interpolation method."""
        params = IDEALParams(ideal_params_file)

        base_array = np.random.rand(4, 4, 4, 4).astype(np.float32)

        # Test with different interpolation method
        result = params.interpolate_start_values(
            base_array, step_idx=0, interpolation=cv2.INTER_LINEAR
        )

        assert result.shape == (8, 8, 4, 4)

    def test_interpolate_start_values_3d_not_implemented(self, ideal_params_file):
        """Test that 3D interpolation raises NotImplementedError."""
        params = IDEALParams(ideal_params_file)
        params.ideal_dims = 3  # Set to 3D

        base_array = np.random.rand(5, 5, 5, 4).astype(np.float32)

        with pytest.raises(
            NotImplementedError, match="Currently only 2D interpolation is supported"
        ):
            params.interpolate_start_values(base_array, step_idx=0)

    def test_interpolate_start_values_various_dtypes(self, ideal_params_file):
        """Test interpolate_start_values with different input dtypes."""
        params = IDEALParams(ideal_params_file)

        # Test with different dtypes
        for dtype in [np.float32, np.float64]:
            base_array = np.random.rand(3, 3, 4, 4).astype(dtype)
            result = params.interpolate_start_values(base_array, step_idx=0)
            assert result.shape == (8, 8, 4, 4)

    def test_interpolate_start_values_edge_cases(self, ideal_params_file):
        """Test interpolate_start_values with edge cases."""
        params = IDEALParams(ideal_params_file)

        # Test with single parameter
        base_array = np.random.rand(2, 2, 1, 6).astype(np.float32)
        result = params.interpolate_start_values(base_array, step_idx=0)
        assert result.shape == (8, 8, 1, 6)

        # Test upscaling and downscaling
        large_array = np.random.rand(50, 50, 4, 6).astype(np.float32)
        result_down = params.interpolate_start_values(large_array, step_idx=0)
        assert result_down.shape == (8, 8, 4, 6)
