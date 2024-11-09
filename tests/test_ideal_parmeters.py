from pyneapple import Parameters, IDEALParams
from .test_toolbox import ParameterTools


def test_init_parameters():
    assert Parameters()


def test_init_ideal_parameters():
    assert IDEALParams()


def test_json_save_ideal(capsys, ideal_params: IDEALParams, out_json):
    # Test IDEAL
    ideal_params.save_to_json(out_json)
    test_params = IDEALParams(out_json)
    attributes = ParameterTools.compare_parameters(ideal_params, test_params)
    ParameterTools.compare_attributes(ideal_params, test_params, attributes)
    capsys.readouterr()
    assert True
