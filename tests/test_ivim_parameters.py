import pytest
import numpy as np
import random
from unittest import mock

from pyneapple import IVIMParams, IVIMSegmentedParams
from pyneapple.parameters import IVIMBoundaries
from radimgarray import SegImgArray
from .test_toolbox import ParameterTools


# @pytest.mark.order(after="test_parameters.py::TestParameters::test_load_b_values")
class TestIVIMParameters:
    def test_init_ivim_parameters(self):
        assert IVIMParams()

    def test_ivim_json_save(self, ivim_tri_params, out_json, capsys):
        # Test IVIM
        ivim_tri_params.save_to_json(out_json)
        test_params = IVIMParams(out_json)
        attributes = ParameterTools.compare_parameters(ivim_tri_params, test_params)
        ParameterTools.compare_attributes(ivim_tri_params, test_params, attributes)
        capsys.readouterr()
        assert True

    def test_eval_fitting_results(self):
        pass
        # TODO: Implement test for eval_fitting_results


class TestIVIMSegmentedParameters:
    def test_init_ivim_segmented_parameters(self, ivim_tri_params_file):
        assert IVIMSegmentedParams(ivim_tri_params_file)

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_valid_fixed_component(self, mock_logger):
        # Preparation: Create a Mock-Boundaries object with necessary data
        params = IVIMSegmentedParams()
        params.fixed_component = "D_slow"
        params.boundaries.dict = {
            "D": {"slow": [0.001, 0.0007, 0.05], "fast": [0.02, 0.003, 0.3]},
            "f": {"slow": [85, 10, 500], "fast": [20, 1, 100]},
        }

        # Patch the load methods to check behavior
        with mock.patch.object(
            params.params_1.boundaries, "load"
        ) as mock_fixed_load, mock.patch.object(
            params.params_2.boundaries, "load"
        ) as mock_boundaries_load:

            # Action: Call _set_up
            params.set_up()

            # Verification: params_fixed.boundaries.load was called with the correct values
            expected_fixed_dict = {
                "D": {"slow": [0.001, 0.0007, 0.05]},
                "f": {"slow": [85, 10, 500]},
            }
            mock_fixed_load.assert_called_once()
            args, _ = mock_fixed_load.call_args
            assert args[0] == expected_fixed_dict

            # Verification: boundaries.load was called and D_slow was removed
            mock_boundaries_load.assert_called_once()
            args, _ = mock_boundaries_load.call_args
            boundary_dict = args[0]
            assert "slow" not in boundary_dict["D"]
            assert "fast" in boundary_dict["D"]

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_invalid_fixed_component(self, mock_logger):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_nonexistent"
        params.boundaries.dict = {
            "D": {"slow": [0.001, 0.0007, 0.05], "fast": [0.02, 0.003, 0.3]}
        }

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
        params.fixed_component = "D_slow"
        params.fixed_t1 = True
        params.fit_t1 = True
        params.mixing_time = 100
        params.boundaries.dict = {
            "D": {"slow": [0.001, 0.0007, 0.05]},
            "f": {"slow": [85, 10, 500]},
            "T": {"t1": [1000, 500, 2000]},
        }

        # Patch the load methods
        with mock.patch.object(
            params.params_1.boundaries, "load"
        ) as mock_fixed_load, mock.patch.object(
            params.boundaries, "load"
        ) as mock_boundaries_load:

            # Action
            params.set_up()

            # Verification: T1 values were transferred to params_fixed
            assert params.params_1.fit_t1 == True
            assert params.params_1.mixing_time == 100
            assert params.fit_t1 == False  # deactivated for second fit

            # Check passed boundary dictionaries
            expected_fixed_dict = {
                "D": {"slow": [0.001, 0.0007, 0.05]},
                "T": {"t1": [1000, 500, 2000]},
            }
            args, _ = mock_fixed_load.call_args
            assert "T" in args[0]
            assert args[0]["T"] == expected_fixed_dict["T"]

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_fixed_t1_without_mixing_time(self, mock_logger):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_slow"
        params.fixed_t1 = True
        params.fit_t1 = True
        params.mixing_time = None  # Missing mixing time value
        params.boundaries.dict = {
            "D": {"slow": [0.001, 0.0007, 0.05]},
            "f": {"slow": [85, 10, 500]},
            "T": {"t1": [1000, 500, 2000]},
        }

        # Action and verification: Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            params.set_up()

        assert "Mixing time is set but not passed" in str(excinfo.value)
        mock_logger.error.assert_called_once()

    @mock.patch("pyneapple.parameters.ivim.logger")
    def test_set_up_fixed_t1_without_t1_boundaries(self, mock_logger):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_slow"
        params.fixed_t1 = True
        params.fit_t1 = True
        params.mixing_time = 100
        params.boundaries.dict = {
            "D": {"slow": [0.001, 0.0007, 0.05]},
            "f": {"slow": [85, 10, 500]},
            # No T-boundaries
        }

        # Action and verification: Should raise KeyError
        with pytest.raises(KeyError) as excinfo:
            params.set_up()

        assert "T1 has no defined boundaries" in str(excinfo.value)
        mock_logger.error.assert_called_once()

    def test_set_up_reduced_b_values(self):
        # Preparation
        params = IVIMSegmentedParams()
        params.fixed_component = "D_slow"
        params.boundaries.dict = {
            "D": {"slow": [0.001, 0.0007, 0.05]},
            "f": {"slow": [85, 10, 500]},
        }
        params.b_values = np.array([0, 10, 20, 30, 40, 50])
        params.reduced_b_values = np.array([0, 30, 50])

        # Patch the load methods
        with mock.patch.object(params.params_1.boundaries, "load"), mock.patch.object(
            params.boundaries, "load"
        ):

            # Action
            params.set_up()

            # Verification: Reduced b-values were passed to params_fixed
            np.testing.assert_array_equal(
                params.params_1.b_values, params.reduced_b_values
            )

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
            assert d_values[pixel_idx] == element[1][1]
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
