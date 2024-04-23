import pytest

from pyneapple.fit import parameters
from test_toolbox import ParameterTools


def test_init_nnls_parameters():
    assert parameters.NNLSParams()


def test_init_nnls_cv_parameters():
    assert parameters.NNLSCVParams()


def test_json_save_nnls(capsys, nnls_params, out_json):
    # Test NNLS
    nnls_params.save_to_json(out_json)
    test_params = parameters.NNLSParams(out_json)
    attributes = ParameterTools.compare_parameters(nnls_params, test_params)
    ParameterTools.compare_attributes(nnls_params, test_params, attributes)
    capsys.readouterr()
    assert True


def test_json_save_nnlscv(capsys, nnlscv_params, out_json):
    # Test NNLS CV
    nnlscv_params.save_to_json(out_json)
    test_params = parameters.NNLSCVParams(out_json)
    attributes = ParameterTools.compare_parameters(nnlscv_params, test_params)
    ParameterTools.compare_attributes(nnlscv_params, test_params, attributes)
    capsys.readouterr()
    assert True
