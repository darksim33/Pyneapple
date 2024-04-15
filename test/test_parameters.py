from pathlib import Path
from pyneapple.fit.parameters import Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams


def test_init_parameters():
    Parameters()
    assert True


def test_init_nnls_parameters():
    NNLSParams()
    assert True


def test_init_nnls_cv_parameters():
    NNLSCVParams()
    assert True


def test_init_ivim_parameters():
    IVIMParams()
    assert True


def test_init_ideal_parameters():
    IDEALParams()
    assert True


def test_json_load():
    IVIMParams().load_from_json(Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json"))
    NNLSParams().load_from_json(Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json"))
    NNLSCVParams().load_from_json(Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json"))
    assert True


def test_json_save(capsys):
    params = IVIMParams(Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json"))
    params.save_to_json(Path(r"test_params.json"))
    params = NNLSParams(Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json"))
    params.save_to_json(Path(r"test_params_nnls.json"))
    params = NNLSCVParams(Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json"))
    params.save_to_json(Path(r"test_params_nnls_reg.json"))
    capsys.readouterr()
    assert True
