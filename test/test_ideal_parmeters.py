import pytest

from pyneapple.fit import parameters
from test_toolbox import ParameterTools


def test_init_parameters():
    assert parameters.Parameters()


def test_init_ideal_parameters():
    assert parameters.IDEALParams()


def test_json_save_ideal(capsys, ideal_params: parameters.IDEALParams, out_json):
    # Test IDEAL
    ideal_params.save_to_json(out_json)
    test_params = parameters.IDEALParams(out_json)
    attributes = ParameterTools.compare_parameters(ideal_params, test_params)
    ParameterTools.compare_attributes(ideal_params, test_params, attributes)
    capsys.readouterr()
    assert True
