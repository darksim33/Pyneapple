import numpy as np
from pathlib import Path
from functools import partial

import pytest

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
    assert Parameters()


def test_init_nnls_parameters():
    assert NNLSParams()


def test_init_nnls_cv_parameters():
    assert NNLSCVParams()


def test_init_ivim_parameters():
    assert IVIMParams()


def test_init_ideal_parameters():
    assert IDEALParams()


def test_json_importer():
    params = JsonImporter(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    )
    isinstance(params, IVIMParams)
    assert True


def test_json_load(ivim_tri_params, nnls_params, nnlscv_params, ideal_params):
    assert True


def test_json_save_ivim(capsys, ivim_tri_params):
    # Test IVIM
    ivim_tri_params.save_to_json(test_json)
    test_params = IVIMParams(test_json)
    attributes = compare_parameters(ivim_tri_params, test_params)
    compare_attributes(ivim_tri_params, test_params, attributes)
    capsys.readouterr()
    assert True


def test_json_save_ideal(capsys, ideal_params):
    # Test IDEAL
    ideal_params.save_to_json(test_json)
    test_params = IDEALParams(test_json)
    attributes = compare_parameters(ideal_params, test_params)
    compare_attributes(ideal_params, test_params, attributes)
    capsys.readouterr()
    assert True


def test_json_save_nnls(capsys, nnls_params):
    # Test NNLS
    nnls_params.save_to_json(test_json)
    test_params = NNLSParams(test_json)
    attributes = compare_parameters(nnls_params, test_params)
    compare_attributes(nnls_params, test_params, attributes)
    capsys.readouterr()
    assert True


def test_json_save_nnlscv(capsys, nnlscv_params):
    # Test NNLS CV
    nnlscv_params.save_to_json(test_json)
    test_params = NNLSCVParams(test_json)
    attributes = compare_parameters(nnlscv_params, test_params)
    compare_attributes(nnlscv_params, test_params, attributes)

    capsys.readouterr()
    assert True


def get_attributes(item) -> list:
    return [
        attr
        for attr in dir(item)
        if not callable(getattr(item, attr))
        and not attr.startswith("_")
        and not isinstance(getattr(item, attr), partial)
    ]


def compare_parameters(params1, params2) -> list:
    """
    Compares two parameter sets.

    Args:
        params1: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
        params2: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams

    Returns:

    """
    # compare attributes first
    attributes = get_attributes(params1)
    test_attributes = get_attributes(params2)

    assert attributes == test_attributes
    return attributes


def compare_attributes(params1, params2, attributes: list):
    """
    Compares attribute values of two parameter sets
    Args:
        params1: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
        params2: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
        attributes:

    Returns:

    """
    for attr in attributes:
        if isinstance(getattr(params1, attr), np.ndarray):
            assert getattr(params1, attr).all() == getattr(params2, attr).all()
        elif attr == "boundaries":
            compare_boundaries(getattr(params1, attr), getattr(params2, attr))
        else:
            assert getattr(params1, attr) == getattr(params2, attr)


def compare_boundaries(boundary1, boundary2):
    attributes1 = get_attributes(boundary1)
    attributes2 = get_attributes(boundary2)

    assert attributes1 == attributes2

    for attr in attributes1:
        if isinstance(getattr(boundary1, attr), np.ndarray):
            assert getattr(boundary1, attr).all() == getattr(boundary2, attr).all()
        else:
            assert getattr(boundary1, attr) == getattr(boundary2, attr)
