import pytest

from pyneapple import NNLSParams, NNLSCVParams
from pyneapple import Results

from .test_toolbox import ParameterTools


class TestNNLSParameters:
    def test_nnls_init_parameters(self):
        assert NNLSParams()

    def test_nnls_get_basis(self, nnls_params):
        basis = nnls_params.get_basis()
        assert basis.shape == (
            nnls_params.boundaries.number_points + nnls_params.b_values.shape[0],
            nnls_params.boundaries.number_points,
        )
        assert basis.max() == 1
        assert basis.min() == 0
        assert True

    def test_nnls_get_pixel_args(self, nnls_params, img, seg):
        args = nnls_params.get_pixel_args(img, seg)
        assert args is not None

    @pytest.mark.parametrize("seg_number", [1, 2, 3])
    def test_nnls_get_seg_args(self, nnls_params, img, seg, seg_number):
        args = nnls_params.get_seg_args(img, seg, seg_number)
        assert args is not None

    def test_nnls_json_save(self, nnls_params, out_json):
        # Test NNLS
        nnls_params.save_to_json(out_json)
        test_params = NNLSParams(out_json)
        attributes = ParameterTools.compare_parameters(nnls_params, test_params)
        ParameterTools.compare_attributes(nnls_params, test_params, attributes)
        assert True

    def test_nnls_load_from_toml(self, nnls_toml_params_file, out_toml):
        nnls_params = NNLSParams(nnls_toml_params_file)
        # Test NNLS with different regularization orders and mu        
        for reg_order in [0, 1, 2, 3]:            
            nnls_params.fit_model.reg_order = reg_order
            mu = reg_order * 0.01 + 0.01
            nnls_params.fit_model.mu = mu
            # Save parameters to TOML file
            nnls_params.save_to_toml(out_toml)
            
            # Load parameters from TOML file
            loaded_params = NNLSParams(out_toml)

            # Compare original and loaded parameters
            attributes = ParameterTools.compare_parameters(nnls_params, loaded_params)
            ParameterTools.compare_attributes(nnls_params, loaded_params, attributes)
            assert loaded_params.fit_model.reg_order == nnls_params.fit_model.reg_order == reg_order
            assert loaded_params.fit_model.mu == nnls_params.fit_model.mu 
            assert True

    # NNLS_CV
    def test_nnls_cv_init_parameters(self):
        assert NNLSCVParams()

    def test_nnlscv_json_save(self, nnlscv_params, out_json):
        # Test NNLS CV
        nnlscv_params.save_to_json(out_json)
        test_params = NNLSCVParams(out_json)
        attributes = ParameterTools.compare_parameters(nnlscv_params, test_params)
        ParameterTools.compare_attributes(nnlscv_params, test_params, attributes)
        assert True
