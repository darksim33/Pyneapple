import numpy as np
from pathlib import Path
from functools import partial
from pyneapple.fit.parameters import (
    Parameters,
    IVIMParams,
    NNLSParams,
    NNLSCVParams,
    IDEALParams,
    JsonImporter,
)

# Settings
test_json = Path(r"test_params.json")


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


def test_json_importer():
    params = JsonImporter(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    )
    isinstance(params, IVIMParams)
    assert True


def test_json_load():
    IVIMParams().load_from_json(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    )
    NNLSParams().load_from_json(
        Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json")
    )
    NNLSCVParams().load_from_json(
        Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json")
    )
    IDEALParams().load_from_json(
        Path(r"../src/pyneapple/resources/fitting/default_params_IDEAL_bi.json")
    )
    assert True


def test_json_save_ivim(capsys):
    # Test IVIM
    params = IVIMParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    )
    params.save_to_json(test_json)
    test_params = IVIMParams(test_json)
    attributes = compare_parameters(params, test_params)
    compare_attributes(params, test_params, attributes)
    capsys.readouterr()
    assert True


def test_json_save_ideal(capsys):
    # Test IDEAL
    params = IDEALParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_IDEAL_tri.json")
    )
    params.save_to_json(test_json)
    test_params = IDEALParams(test_json)
    attributes = compare_parameters(params, test_params)
    compare_attributes(params, test_params, attributes)
    capsys.readouterr()
    assert True


def test_json_save_nnls(capsys):
    # Test NNLS
    params = NNLSParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json")
    )
    params.save_to_json(test_json)
    test_params = NNLSParams(test_json)
    attributes = compare_parameters(params, test_params)
    compare_attributes(params, test_params, attributes)
    capsys.readouterr()
    assert True


def test_json_save_nnlscv(capsys):
    # Test NNLS CV
    params = NNLSCVParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json")
    )
    params.save_to_json(test_json)
    test_params = NNLSCVParams(test_json)
    attributes = compare_parameters(params, test_params)
    compare_attributes(params, test_params, attributes)

    capsys.readouterr()
    assert True


def compare_parameters(params1: IVIMParams, params2: IVIMParams) -> list:
    # compare attributes first
    attributes = [
        attr
        for attr in dir(params1)
        if not callable(getattr(params1, attr))
        and not attr.startswith("_")
        and not isinstance(getattr(params1, attr), partial)
    ]
    test_attributes = [
        attr
        for attr in dir(params2)
        if not callable(getattr(params2, attr))
        and not attr.startswith("_")
        and not isinstance(getattr(params2, attr), partial)
    ]

    if not attributes == test_attributes:
        raise ValueError(f"Parameters attributes do not match!")

    return attributes


def compare_attributes(params1: IVIMParams, params2: IVIMParams, attributes: list):
    for attr in attributes:
        if isinstance(getattr(params1, attr), np.ndarray):
            if not getattr(params1, attr).all() == getattr(params2, attr).all():
                ValueError(f"{attr} is not a valid parameter")
        elif not getattr(params1, attr) == getattr(params2, attr):
            ValueError(f"{attr} is not a valid parameter")
