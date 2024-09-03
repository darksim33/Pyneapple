import pytest
import numpy as np
import random

from pyneapple.fit import parameters
from pyneapple.fit.results import Results
from test_toolbox import ParameterTools


# @pytest.mark.order(after="test_parameters.py::TestParameters::test_load_b_values")
class TestIVIMParameters:
    def test_init_ivim_parameters(self):
        assert parameters.IVIMParams()

    def test_ivim_json_save(self, ivim_tri_params, out_json, capsys):
        # Test IVIM
        ivim_tri_params.save_to_json(out_json)
        test_params = parameters.IVIMParams(out_json)
        attributes = ParameterTools.compare_parameters(ivim_tri_params, test_params)
        ParameterTools.compare_attributes(ivim_tri_params, test_params, attributes)
        capsys.readouterr()
        assert True

    def test_ivim_boundaries(self, ivim_tri_params, capsys):
        bins = ivim_tri_params.get_bins()
        assert [round(min(bins), 5), round(max(bins), 5)] == [
            min(ivim_tri_params.boundaries.dict["D"]["slow"]),
            max(ivim_tri_params.boundaries.dict["D"]["fast"]),
        ]


class TestIVIMSegmentedParameters:
    def test_init_ivim_segmented_parameters(self, ivim_tri_params_file):
        assert parameters.IVIMSegmentedParams(ivim_tri_params_file)

    @pytest.fixture
    def fixed_values(self, seg):
        shape = np.squeeze(seg.array).shape
        d_slow_map = np.random.rand(*shape)
        t1_map = np.random.randint(1, 2500, shape)
        indexes = list(np.ndindex(shape))
        d_slow, t1 = {}, {}
        for idx in indexes:
            d_slow[idx] = d_slow_map[idx]
            t1[idx] = t1_map[idx]

        return d_slow, t1

    @pytest.fixture
    def results_bi_exp(self, seg):
        shape = np.squeeze(seg.array).shape
        d_fast_map = np.random.rand(*shape)
        f_map = np.random.rand(*shape)
        s_0_map = np.random.randint(1, 2500, shape)
        indexes = list(np.ndindex(shape))
        results = []
        for idx in indexes:
            results.append((idx, np.array([d_fast_map[idx], f_map[idx], s_0_map[idx]])))

        return results

    def test_set_options(self, ivim_tri_t1_params_file):
        # Preparing dummy Mono params
        dummy_params = parameters.IVIMParams(ivim_tri_t1_params_file)
        dummy_params.TM = 100

        # Setting Options for segmented fitting
        params = parameters.IVIMSegmentedParams(
            ivim_tri_t1_params_file,
        )

        assert not params.params_fixed.scale_image

        params.TM = 100
        params.scale_image = "S/S0"
        params.set_options(
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 500],
        )
        assert params.params_fixed.scale_image == "S/S0"

        assert (
            params.params_fixed.boundaries.dict["D"]["slow"]
            == dummy_params.boundaries.dict["D"]["slow"]
        )
        assert params.params_fixed.TM == dummy_params.TM
        assert not params.TM
        assert params.params_fixed.boundaries.dict.get("T", None) is not None
        assert params.boundaries.dict.get("T", False) is False

    def test_get_fixed_fit_results(self, ivim_tri_params_file, fixed_results):
        params = parameters.IVIMSegmentedParams(
            ivim_tri_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 500],
        )
        d_values, t1_values = params.get_fixed_fit_results(
            fixed_results, shape=(2, 2, 1)
        )
        assert d_values == fixed_results.d
        assert t1_values == fixed_results.T1

    def test_get_pixel_args_fixed(self, img, seg, ivim_tri_params_file):
        params = parameters.IVIMSegmentedParams(
            ivim_tri_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 50, 550, 650],
        )
        pixel_args = params.get_pixel_args_fixed(img, seg)
        for arg in pixel_args:
            assert len(arg[1]) == len(params.options["reduced_b_values"])

    def test_get_pixel_args(self, img, seg, ivim_tri_params_file):
        params = parameters.IVIMSegmentedParams(
            ivim_tri_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 50, 550, 650],
        )
        adc = np.squeeze(np.random.randint(1, 2500, seg.array.shape))
        t1 = np.squeeze((np.random.randint(1, 2000, seg.array.shape)))
        fixed_values = [adc, t1]
        pixel_args = params.get_pixel_args(img, seg, *fixed_values)
        for arg in pixel_args:
            assert arg[2] == fixed_values[0][*arg[0]]
            assert arg[3] == fixed_values[1][*arg[0]]

    def test_eval_fitting_results_bi_exp(
        self, ivim_bi_params_file, results_bi_exp, fixed_values
    ):
        params = parameters.IVIMSegmentedParams(
            ivim_bi_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 50, 550, 650],
        )
        results_dict = params.eval_fitting_results(
            results_bi_exp, fixed_component=fixed_values
        )
        # test D_slow
        for key in fixed_values[0]:
            assert results_dict["d"][key][0] == fixed_values[0][key]

        for element in results_bi_exp:
            pixel_idx = element[0]
            assert results_dict["S0"][pixel_idx] == element[1][-1]
            assert results_dict["f"][pixel_idx][0] == element[1][1]
            assert results_dict["f"][pixel_idx][1] >= 1 - element[1][1]
            assert results_dict["d"][pixel_idx][1] == element[1][0]
