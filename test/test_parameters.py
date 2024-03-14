import pytest
from pathlib import Path
from src.fit.parameters import Parameters, NNLSParams, NNLSregParams


def test_json_save(capsys):
    params = Parameters(Path(r"resources/fitting/default_params_IVIM_tri.json"))
    params.save_to_json(Path(r"test_params.json"))
    params = NNLSParams(Path(r"resources/fitting/default_params_NNLSreg.json"))
    params.save_to_json(Path(r"test_params_nnls.json"))
    params = NNLSregParams(Path(r"resources/fitting/default_params_NNLSreg.json"))
    params.save_to_json(Path(r"test_params_nnls_reg.json"))
    capsys.readouterr()
    assert True


def test_json_load():
    Parameters().load_from_json(Path(r"resources/fitting/default_params_IVIM_tri.json"))
    Parameters().load_from_json(Path(r"resources/fitting/default_params_NNLSreg.json"))
    Parameters().load_from_json(Path(r"resources/fitting/default_params_NNLSreg.json"))
    assert True
