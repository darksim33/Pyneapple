import pytest
import numpy as np
from unittest import mock

from pyneapple import IVIMParams, IVIMSegmentedParams
from pyneapple.parameters import IVIMBoundaryDict
from pyneapple.models import MonoExpFitModel, BiExpFitModel, TriExpFitModel
from radimgarray import SegImgArray
from .test_toolbox import ParameterTools


# @pytest.mark.order(after="test_parameters.py::TestParameters::test_load_b_values")
class TestIVIMParameters:
    def test_init_ivim_parameters(self):
        """Test basic initialization of IVIMParams."""
        params = IVIMParams()
        assert isinstance(params, IVIMParams)
        assert isinstance(params.boundaries, IVIMBoundaryDict)

    def test_init_with_file(self, ivim_tri_params_file):
        """Test initialization with parameter file."""
        params = IVIMParams(ivim_tri_params_file)
        assert isinstance(params, IVIMParams)

    def test_init_with_t1_but_no_mixing_time(self, ivim_tri_t1_no_mixing_params_file):
        """Test that initialization fails when T1 is enabled but no mixing time is set."""
        with pytest.raises(
            ValueError, match="T1 mapping is set but no mixing time is defined"
        ):
            IVIMParams(ivim_tri_t1_no_mixing_params_file)

    def test_model_setter_mono_exponential(self):
        """Test setting mono-exponential model."""
        params = IVIMParams()
        params._set_model("MonoExp")

        assert params.fit_model.name == "MonoExp"
        assert isinstance(params._fit_model, MonoExpFitModel)

    def test_model_setter_bi_exponential(self):
        """Test setting bi-exponential model."""
        params = IVIMParams()
        params._set_model("BiExp")

        assert params.fit_model.name == "BiExp"
        assert isinstance(params._fit_model, BiExpFitModel)

    def test_model_setter_tri_exponential(self):
        """Test setting tri-exponential model."""
        params = IVIMParams()
        params._set_model("TriExp")

        assert params.fit_model.name == "TriExp"
        assert isinstance(params._fit_model, TriExpFitModel)

    def test_model_setter_invalid_model(self):
        """Test that setting invalid model raises error."""
        params = IVIMParams()

        with pytest.raises(ValueError, match="Only exponential models are supported"):
            params._set_model("InvalidModel")

    def test_model_setter_unsupported_exponential(self):
        """Test that unsupported exponential models raise error."""
        params = IVIMParams()

        with pytest.raises(
            ValueError, match="Only mono-, bi- and tri-exponential models are supported"
        ):
            params._set_model("QuadExp")

    def test_fit_model_property_with_reduced(self):
        """Test fit_model property with fit_reduced flag."""
        params = IVIMParams()
        params._set_model("BiExp")
        params.fit_model.fit_reduced = True

        assert params.fit_model.fit_reduced == True

    def test_fit_model_property_with_t1(self):
        """Test fit_model property with T1 fitting."""
        params = IVIMParams()
        params._set_model("BiExp")
        params.fit_model.fit_t1 = True
        params.fit_model.mixing_time = 100

        assert params.fit_model.fit_t1 == True
        assert params.fit_model.mixing_time == 100

    def test_fit_model_property_with_s0(self):
        """Test fit_model property with S0 fitting."""
        params = IVIMParams()
        params._set_model("BiExp")
        params.fit_model.fit_S0 = True

        assert params.fit_model.fit_S0 == True

    def test_fit_function_property(self):
        """Test fit_function property returns partial function."""
        params = IVIMParams()
        params._set_model("MonoExp")
        params.b_values = np.array([0, 10, 20, 50, 100])
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0007, 0.05]},
                "S": {"0": [85, 10, 500]},
            }
        )

        fit_func = params.fit_function
        assert callable(fit_func)

    def test_fit_function_general_boundaries(self):
        """Test fit_function with general boundaries."""
        params = IVIMParams()
        params._set_model("MonoExp")
        params.b_values = np.array([0, 10, 20, 50, 100])
        params.boundaries = IVIMBoundaryDict(
            {"D": {"1": [0.001, 0.0007, 0.05]}, "S": {"0": [85, 10, 500]}}
        )

        fit_func = params.fit_function

        # Verify it's a partial function
        assert callable(fit_func)
        assert hasattr(fit_func, "func")

        # Verify the partial has the correct parameters set
        assert "b_values" in fit_func.keywords
        assert "x0" in fit_func.keywords
        assert "lb" in fit_func.keywords
        assert "ub" in fit_func.keywords
        assert "max_iter" in fit_func.keywords

    def test_fit_function_individual_boundaries(self):
        """Test fit_function with individual boundaries."""
        params = IVIMParams()
        params._set_model("MonoExp")
        params.b_values = np.array([0, 10, 20, 50, 100])

        shape_3d = (2, 2, 2)
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {
                    "1": {
                        (0, 0, 0): np.full(shape_3d, 0.001),
                        (0, 0, 1): np.full(shape_3d, 0.0007),
                        (0, 1, 0): np.full(shape_3d, 0.05),
                    }
                },
                "S": {
                    "0": {
                        (0, 0, 0): np.full(shape_3d, 85),
                        (0, 0, 1): np.full(shape_3d, 10),
                        (0, 1, 0): np.full(shape_3d, 500),
                    }
                },
            }
        )

        fit_func = params.fit_function

        # Verify it's a partial function
        assert callable(fit_func)
        assert hasattr(fit_func, "func")

        # Verify the partial has the correct parameters set
        assert "b_values" in fit_func.keywords
        assert "max_iter" in fit_func.keywords
        # For pixel-wise fitting, boundaries are passed per-pixel, not in partial
        assert "x0" not in fit_func.keywords
        assert "lb" not in fit_func.keywords
        assert "ub" not in fit_func.keywords

    def test_get_basis(self):
        """Test get_basis method."""
        params = IVIMParams()
        b_values = np.array([0, 10, 20, 50, 100])
        params.b_values = b_values

        basis = params.get_basis()
        np.testing.assert_array_equal(basis, b_values.squeeze())

    def test_normalize_static_method(self):
        """Test normalize static method."""
        # Create test image with known values
        img = np.zeros((2, 2, 2, 4))
        img[0, 0, 0, :] = [100, 80, 60, 40]  # S0=100
        img[1, 1, 1, :] = [200, 160, 120, 80]  # S0=200

        normalized = IVIMParams.normalize(img)

        # Check normalization
        expected_00 = np.array([1.0, 0.8, 0.6, 0.4])
        expected_11 = np.array([1.0, 0.8, 0.6, 0.4])

        np.testing.assert_array_almost_equal(normalized[0, 0, 0, :], expected_00)
        np.testing.assert_array_almost_equal(normalized[1, 1, 1, :], expected_11)

        # Check that zero voxels remain zero
        np.testing.assert_array_equal(normalized[0, 1, 0, :], np.zeros(4))

    def test_ivim_json_save_and_load(self, ivim_tri_params, out_json):
        """Test saving and loading IVIM parameters to/from JSON."""
        # Save parameters
        ivim_tri_params.save_to_json(out_json)

        # Load parameters
        test_params = IVIMParams(out_json)

        # Compare parameters
        attributes = ParameterTools.compare_parameters(ivim_tri_params, test_params)
        ParameterTools.compare_attributes(ivim_tri_params, test_params, attributes)

    def test_get_pixel_args_general_boundaries(self, img, seg):
        """Test get_pixel_args with general boundary type."""
        params = IVIMParams()
        params._set_model("BiExp")
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0007, 0.05], "2": [0.02, 0.003, 0.3]},
                "f": {"1": [85, 10, 500], "2": [20, 1, 100]},
            }
        )

        pixel_args = list(params.get_pixel_args(img, seg))

        # Verify we have the correct number of pixels
        assert len(pixel_args) == np.count_nonzero(seg)

        # Verify each argument tuple has correct structure
        for arg in pixel_args:
            assert len(arg) == 2  # (coordinates, signal)
            assert len(arg[0]) == 3  # (i, j, k)
            assert len(arg[1]) == img.shape[-1]  # signal length matches b-values

    def test_get_pixel_args_individual_boundaries(self, img, seg):
        """Test get_pixel_args with individual/pixel-wise boundary type."""
        params = IVIMParams()
        params._set_model("BiExp")
        params.b_values = np.array([0, 10, 20, 50, 100])

        # Generate random pixel-wise boundaries for all segmented pixels
        seg_indices = list(
            tuple(map(int, (i, j, k)))
            for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
        )

        d1_bounds = {}
        d2_bounds = {}
        f1_bounds = {}
        f2_bounds = {}

        for idx in seg_indices:
            # D1: x0 between 0.0005-0.002, lb slightly lower, ub slightly higher
            d1_x0 = np.random.uniform(0.0005, 0.002)
            d1_bounds[idx] = [d1_x0, d1_x0 * 0.5, d1_x0 * 2.0]

            # D2: x0 between 0.01-0.05, lb slightly lower, ub slightly higher
            d2_x0 = np.random.uniform(0.01, 0.05)
            d2_bounds[idx] = [d2_x0, d2_x0 * 0.5, d2_x0 * 2.0]

            # f1: x0 between 50-100, lb at 10, ub at 500
            f1_x0 = np.random.uniform(50, 100)
            f1_bounds[idx] = [f1_x0, 10, 500]

            # f2: x0 between 10-30, lb at 1, ub at 100
            f2_x0 = np.random.uniform(10, 30)
            f2_bounds[idx] = [f2_x0, 1, 100]

        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": d1_bounds, "2": d2_bounds},
                "f": {"1": f1_bounds, "2": f2_bounds},
            }
        )

        pixel_args = list(params.get_pixel_args(img, seg))

        # Verify we have the correct number of pixels
        assert len(pixel_args) == np.count_nonzero(seg)

        # Verify each argument tuple has correct structure for pixel-wise fitting
        for arg in pixel_args:
            assert len(arg) == 5  # (coordinates, signal, x0, lb, ub)
            assert len(arg[0]) == 3  # (i, j, k)
            assert len(arg[1]) == img.shape[-1]  # signal length
            assert len(arg[2]) == 4  # x0 for BiExp: [D1, D2, f1, f2]
            assert len(arg[3]) == 4  # lower bounds
            assert len(arg[4]) == 4  # upper bounds

    def test_get_pixel_args_invalid_boundary_type(self, img, seg):
        """Test that invalid boundary type raises error."""
        params = IVIMParams()
        with pytest.raises(ValueError, match="Boundary type .* not recognized"):
            list(params.get_pixel_args(img, seg))


class TestIVIMSegmentedParameters:
    def test_init_ivim_segmented_parameters(self, ivim_tri_params_file):
        assert IVIMSegmentedParams(ivim_tri_params_file)

    # Basic initialization tests (inherited from IVIMParams)
    def test_init_basic_properties(self):
        """Test basic initialization properties inherited from IVIMParams."""
        params = IVIMSegmentedParams()
        assert isinstance(params, IVIMSegmentedParams)
        assert isinstance(params, IVIMParams)  # inheritance check
        assert isinstance(params.boundaries, IVIMBoundaryDict)
        assert not params.fit_model.fit_reduced
        assert not params.fit_model.fit_t1
        assert params.fit_model.mixing_time is None

        # Additional segmented-specific properties
        assert params.fixed_component == ""
        assert not params.fixed_t1
        assert isinstance(params.params_1, IVIMParams)
        assert isinstance(params.params_2, IVIMParams)

    def test_init_with_file(self, ivim_tri_params_file):
        """Test initialization with parameter file."""
        params = IVIMSegmentedParams(ivim_tri_params_file)
        assert isinstance(params, IVIMSegmentedParams)

    def test_init_with_t1_but_no_mixing_time(self, ivim_tri_t1_no_mixing_params_file):
        """Test that initialization fails when T1 is enabled but no mixing time is set."""
        with pytest.raises(
            ValueError, match="T1 mapping is set but no mixing time is defined"
        ):
            IVIMSegmentedParams(ivim_tri_t1_no_mixing_params_file)

    # Model setter tests (inherited behavior)
    def test_model_setter_mono_exponential(self):
        """Test setting mono-exponential model."""
        params = IVIMSegmentedParams()
        params._set_model("MonoExp")

        assert params.fit_model.name == "MonoExp"
        assert isinstance(params._fit_model, MonoExpFitModel)

    def test_model_setter_bi_exponential(self):
        """Test setting bi-exponential model."""
        params = IVIMSegmentedParams()
        params._set_model("BiExp")

        assert params.fit_model.name == "BiExp"
        assert isinstance(params._fit_model, BiExpFitModel)

    def test_model_setter_tri_exponential(self):
        """Test setting tri-exponential model."""
        params = IVIMSegmentedParams()
        params._set_model("TriExp")

        assert params.fit_model.name == "TriExp"
        assert isinstance(params._fit_model, TriExpFitModel)

    def test_model_setter_invalid_model(self):
        """Test that setting invalid model raises error."""
        params = IVIMSegmentedParams()

        with pytest.raises(ValueError, match="Only exponential models are supported"):
            params.model = "InvalidModel"

    def test_model_setter_unsupported_exponential(self):
        """Test that unsupported exponential models raise error."""
        params = IVIMSegmentedParams()

        with pytest.raises(
            ValueError, match="Only mono-, bi- and tri-exponential models are supported"
        ):
            params.model = "QuadExp"

    def test_segmented_json_save_and_load(self, ivim_tri_params, out_json):
        """Test saving and loading segmented IVIM parameters to/from JSON."""
        # Create segmented parameters with some specific settings
        seg_params = IVIMSegmentedParams()
        seg_params.fit_type = "single"
        seg_params.b_values = ivim_tri_params.b_values
        seg_params.boundaries = ivim_tri_params.boundaries
        seg_params.fit_model = ivim_tri_params.fit_model
        seg_params.fixed_component = "D_1"
        seg_params.fixed_t1 = False
        seg_params.reduced_b_values = np.array([0, 50, 100])
        seg_params.set_up()

        # Save parameters
        seg_params.save_to_json(out_json)

        # Load parameters
        test_params = IVIMSegmentedParams(out_json)

        # Compare basic parameters
        attributes = ParameterTools.compare_parameters(seg_params, test_params)
        ParameterTools.compare_attributes(seg_params, test_params, attributes)

    # Segmented-specific property tests
    def test_fixed_component_setter_valid(self):
        """Test setting valid fixed component."""
        params = IVIMSegmentedParams()
        params.fixed_component = "D_slow"
        assert params.fixed_component == "D_slow"

        params.fixed_component = "D_1"
        assert params.fixed_component == "D_1"

    def test_fixed_component_setter_invalid_format(self):
        """Test setting invalid fixed component format."""
        params = IVIMSegmentedParams()

        with pytest.raises(ValueError, match="Fixed component must be in the form"):
            params.fixed_component = "D_slow_invalid"

        with pytest.raises(ValueError, match="Fixed component must be in the form"):
            params.fixed_component = "Dslow"

    def test_fixed_component_setter_invalid_type(self):
        """Test setting invalid fixed component type."""
        params = IVIMSegmentedParams()

        with pytest.raises(TypeError, match="Fixed component must be a string"):
            params.fixed_component = 123

    def test_fixed_t1_setter_valid(self):
        """Test setting valid fixed_t1 values."""
        params = IVIMSegmentedParams()
        params.fixed_t1 = True
        assert params.fixed_t1

        params.fixed_t1 = False
        assert not params.fixed_t1

    def test_fixed_t1_setter_invalid(self):
        """Test setting invalid fixed_t1 values."""
        params = IVIMSegmentedParams()

        with pytest.raises(TypeError, match="Fixed T1 must be a boolean value"):
            params.fixed_t1 = "true"

    def test_reduced_b_values_setter_list(self):
        """Test setting fit_reduced b_values with list."""
        params = IVIMSegmentedParams()
        b_vals = [0, 50, 100, 200]
        params.reduced_b_values = b_vals

        expected = np.expand_dims(np.array(b_vals), axis=1)
        np.testing.assert_array_equal(params.reduced_b_values, expected)

    def test_reduced_b_values_setter_array(self):
        """Test setting fit_reduced b_values with numpy array."""
        params = IVIMSegmentedParams()
        b_vals = np.array([0, 50, 100, 200])
        params.reduced_b_values = b_vals

        expected = np.expand_dims(b_vals, axis=1)
        np.testing.assert_array_equal(params.reduced_b_values, expected)

    def test_reduced_b_values_setter_none(self):
        """Test setting fit_reduced b_values to None."""
        params = IVIMSegmentedParams()
        params.reduced_b_values = None

        expected = np.array([])
        np.testing.assert_array_equal(params.reduced_b_values, expected)

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_valid_fixed_component(self, mock_logger):
        # Preparation: Create a Mock-Boundaries object with necessary data
        params = IVIMSegmentedParams()
        params.fixed_component = "D_1"
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0007, 0.05], "2": [0.02, 0.003, 0.3]},
                "f": {"1": [85, 10, 500], "2": [20, 1, 100]},
            }
        )

        # Action: Call set_up
        params.set_up()

        # Verification: params_1.boundaries should contain the fixed component
        assert "D" in params.params_1.boundaries
        assert "1" in params.params_1.boundaries["D"]
        assert params.params_1.boundaries["D"]["1"] == [0.001, 0.0007, 0.05]

        # Verification: S0 should be added to params_1.boundaries
        assert "S" in params.params_1.boundaries
        assert "0" in params.params_1.boundaries["S"]

        # Verification: params_2.boundaries should not contain the fixed component
        assert "D" in params.params_2.boundaries
        assert "1" not in params.params_2.boundaries["D"]
        assert "2" in params.params_2.boundaries["D"]
        assert params.params_2.boundaries["D"]["2"] == [0.02, 0.003, 0.3]

        # Verification: f values should be in params_2.boundaries
        assert "f" in params.params_2.boundaries
        assert "2" in params.params_2.boundaries["f"]

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_invalid_fixed_component(self, mock_logger):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_nonexistent"
        params.boundaries = IVIMBoundaryDict(
            {"D": {"1": [0.001, 0.0007, 0.05], "2": [0.02, 0.003, 0.3]}}
        )
        # Action and verification: Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            params.set_up()

        # Check error message
        assert "Fixed component D_nonexistent is not valid" in str(excinfo.value)
        mock_logger.error.assert_called_once()

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_with_fixed_t1(self, mock_logger):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_1"
        params.fixed_t1 = True
        params.fit_model.fit_t1 = True
        params.fit_model.mixing_time = 100
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0007, 0.05]},
                "f": {"1": [85, 10, 500]},
                "T": {"t1": [1000, 500, 2000]},
            }
        )

        # Action

        params.set_up()

        # Verification: T1 values were transferred to params_1
        assert params.params_1.fit_model.fit_t1 == True
        assert params.params_1.fit_model.mixing_time == 100
        assert params.params_2.fit_model.fit_t1 == False

        # Verification: T1 boundaries should be in params_1
        assert "T" in params.params_1.boundaries
        assert "t1" in params.params_1.boundaries["T"]
        assert params.params_1.boundaries["T"]["t1"] == [1000, 500, 2000]

        # Verification: T1 boundaries should not be in params_2
        assert "T" not in params.params_2.boundaries

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_fixed_t1_without_mixing_time(self, mock_logger):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_1"
        params.fixed_t1 = True
        params.fit_model.fit_t1 = True
        params.fit_model.mixing_time = None  # Missing mixing time value
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0007, 0.05]},
                "f": {"1": [85, 10, 500]},
                "T": {"t1": [1000, 500, 2000]},
            }
        )

        # Action and verification: Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            params.set_up()

        assert "Mixing time is set but not passed" in str(excinfo.value)
        mock_logger.error.assert_called_once()

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_fixed_t1_without_t1_boundaries(self, mock_logger):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_1"
        params.fixed_t1 = True
        params.fit_model.fit_t1 = True
        params.fit_model.mixing_time = 100
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0007, 0.05]},
                "f": {"1": [85, 10, 500]},
                # No T-boundaries
            }
        )

        # Action and verification: Should raise KeyError
        with pytest.raises(KeyError) as excinfo:
            params.set_up()

        assert "T1 has no defined boundaries" in str(excinfo.value)
        mock_logger.error.assert_called_once()

    def test_set_up_reduced_b_values(self):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_1"
        params.boundaries = IVIMBoundaryDict(
            {
                "D": {"1": [0.001, 0.0007, 0.05]},
                "f": {"1": [85, 10, 500]},
            }
        )
        params.b_values = np.array([0, 10, 20, 30, 40, 50])
        params.reduced_b_values = np.array([0, 30, 50])

        # Action
        params.set_up()

        # Verification: Reduced b-values were passed to params_1
        np.testing.assert_array_equal(params.params_1.b_values, params.reduced_b_values)

    @pytest.fixture
    def fixed_results(self):
        shape = (2, 2, 1)
        f_slow_map = np.random.randint(1, 2500, shape)
        d_slow_map = np.random.rand(*shape)
        t1_map = np.random.randint(1, 2500, shape)
        indexes = list(np.ndindex(shape))
        fit_results = []
        for idx in indexes:
            fit_results.append(
                (idx, np.array([f_slow_map[idx], d_slow_map[idx], t1_map[idx]]))
            )

        return fit_results

    def test_get_fixed_fit_results(self, ivim_tri_t1_params_file, fixed_results):
        params = IVIMSegmentedParams(
            ivim_tri_t1_params_file,
        )
        params.fixed_component = "D_1"
        params.fixed_t1 = True
        params.reduced_b_values = [0, 500]
        params.set_up()
        d_values, t1_values = params.get_fixed_fit_results(fixed_results)
        for element in fixed_results:
            pixel_idx = element[0]
            assert d_values[pixel_idx] == element[1][0]
            assert t1_values[pixel_idx] == element[1][2]

    def test_get_pixel_args_fit1(self, img, seg, ivim_tri_t1_params_file):
        params = IVIMSegmentedParams(
            ivim_tri_t1_params_file,
        )
        params.fixed_component = "D_1"
        params.fixed_t1 = True
        params.reduced_b_values = [0, 50, 550, 650]
        pixel_args = params.get_pixel_args_fit1(img, seg)
        for arg in pixel_args:
            assert len(arg[1]) == len(params.reduced_b_values)

    def test_get_pixel_args_fit2(self, img, seg, ivim_tri_t1_params_file):
        params = IVIMSegmentedParams(
            ivim_tri_t1_params_file,
        )
        params.fixed_component = "D_1"
        params.fixed_t1 = True
        params.reduced_b_values = [0, 50, 550, 650]
        params.set_up()

        adc = np.squeeze(np.random.randint(1, 2500, seg.shape))
        t1 = np.squeeze((np.random.randint(1, 2000, seg.shape)))
        fixed_values = [adc, t1]
        pixel_args = params.get_pixel_args_fit2(img, seg, *fixed_values)
        for arg in pixel_args:
            assert arg[2] == fixed_values[0][tuple(arg[0])]  # python 3.9 support
            assert arg[3] == fixed_values[1][tuple(arg[0])]
