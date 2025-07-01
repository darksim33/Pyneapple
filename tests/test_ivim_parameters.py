import pytest
import numpy as np
import random

from pyneapple import IVIMParams, IVIMSegmentedParams
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

    def test_set_options(self, ivim_tri_t1_params_file):
        # Preparing dummy Mono params
        dummy_params = IVIMParams(ivim_tri_t1_params_file)
        dummy_params.mixing_time = 100

        # Setting Options for segmented fitting
        params = IVIMSegmentedParams(
            ivim_tri_t1_params_file,
        )

        params.mixing_time = 100
        params.set_options(
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 500],
        )

        assert (
            params.params_fixed.boundaries.dict["D"]["slow"]
            == dummy_params.boundaries.dict["D"]["slow"]
        )
        assert params.params_fixed.mixing_time == dummy_params.mixing_time
        assert not params.mixing_time
        assert params.params_fixed.boundaries.dict.get("T", None) is not None
        assert params.boundaries.dict.get("T", False) is False

    def test_get_fixed_fit_results(self, ivim_tri_t1_params_file, fixed_results):
        params = IVIMSegmentedParams(
            ivim_tri_t1_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 500],
        )
        d_values, t1_values = params.get_fixed_fit_results(fixed_results)
        for element in fixed_results:
            pixel_idx = element[0]
            assert d_values[pixel_idx] == element[1][1]
            assert t1_values[pixel_idx] == element[1][2]

    def test_get_pixel_args_fixed(self, img, seg, ivim_tri_t1_params_file):
        params = IVIMSegmentedParams(
            ivim_tri_t1_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 50, 550, 650],
        )
        pixel_args = params.get_pixel_args_fixed(img, seg)
        for arg in pixel_args:
            assert len(arg[1]) == len(params.options["reduced_b_values"])

    def test_get_pixel_args(self, img, seg, ivim_tri_t1_params_file):
        params = IVIMSegmentedParams(
            ivim_tri_t1_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 50, 550, 650],
        )
        adc = np.squeeze(np.random.randint(1, 2500, seg.shape))
        t1 = np.squeeze((np.random.randint(1, 2000, seg.shape)))
        fixed_values = [adc, t1]
        pixel_args = params.get_pixel_args(img, seg, *fixed_values)
        for arg in pixel_args:
            assert arg[2] == fixed_values[0][tuple(arg[0])]  # python 3.9 support
            assert arg[3] == fixed_values[1][tuple(arg[0])]
