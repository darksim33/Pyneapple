import pytest
from pathlib import Path
from src.fit.parameters import Parameters, NNLSParams, NNLSregParams


def test_json_save(capsys):
    params = Parameters()
    params.save_to_json(Path(r"test_params.json"))
    params = NNLSParams()
    params.save_to_json(Path(r"test_params_nnls.json"))
    params = NNLSregParams()
    params.save_to_json(Path(r"test_params_nnls_reg.json"))
    capsys.readouterr()
    assert True


def test_json_load():
    Parameters.load_from_json(Path(r"test_params.json"))
    Parameters.load_from_json(Path(r"test_params_nnls.json"))
    Parameters.load_from_json(Path(r"test_params_nnls_reg.json"))
    assert True
