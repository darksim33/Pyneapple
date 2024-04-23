import pytest
from pyneapple.fit import parameters
from test_toolbox import ParameterTools


def test_init_ivim_parameters():
    assert parameters.IVIMParams()


def test_json_save_ivim(capsys, ivim_tri_params, out_json):
    # Test IVIM
    ivim_tri_params.save_to_json(out_json)
    test_params = parameters.IVIMParams(out_json)
    attributes = ParameterTools.compare_parameters(ivim_tri_params, test_params)
    ParameterTools.compare_attributes(ivim_tri_params, test_params, attributes)
    capsys.readouterr()
    assert True
