import random

import numpy as np
import pytest
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
    def test_init_ivim_parameters(self):
        assert parameters.IVIMFixedComponentParams()

    def test_init_ivim_segmented_parameters(self, ivim_tri_params_file):
        assert parameters.IVIMSegmentedParams(ivim_tri_params_file)

    @pytest.fixture
    def fixed_parameters(self):
        shape = (2, 2, 2)
        d_slow_map = np.random.rand(*shape)
        t1_map = np.random.randint(1, 2500, shape)
        return d_slow_map, t1_map

    @pytest.fixture
    def fixed_results(self):
        d_values = {
            (0, 0, 0): random.random(),
            (0, 1, 0): random.random(),
            (1, 0, 0): random.random(),
            (1, 1, 0): random.random(),
        }
        t1_values = {
            (0, 0, 0): random.random() * 1000,
            (0, 1, 0): random.random() * 1000,
            (1, 0, 0): random.random() * 1000,
            (1, 1, 0): random.random() * 1000,
        }

        results = {
            "d": d_values,
            "T1": t1_values,
        }

        fit_results = Results()
        fit_results.update_results(results)
        return fit_results

    def test_set_options(self, ivim_tri_params_file):
        # Preparing dummy Mono params
        dummy_params = parameters.IVIMParams(ivim_tri_params_file)
        dummy_params.TM = 100

        # Setting Options for segmented fitting
        params = parameters.IVIMSegmentedParams(
            ivim_tri_params_file,
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
        assert d_values.shape == (2, 2, 1)
        assert t1_values.shape == (2, 2, 1)

    # def test_get_pixel_args(self, img, seg_reduced, fixed_parameters):
    #     params = parameters.IVIMFixedComponentParams()
    #     params.get_pixel_args(img, seg_reduced, *fixed_parameters)
