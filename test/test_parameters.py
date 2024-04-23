import numpy as np
from pathlib import Path
from functools import partial
from time import sleep

import pytest

from pyneapple.fit.parameters import (
    Parameters,
    IVIMParams,
    NNLSParams,
    NNLSCVParams,
    IDEALParams,
    JsonImporter,
)


def test_init_parameters():
    assert Parameters()


def test_init_ideal_parameters():
    assert IDEALParams()


def test_json_importer():
    params = JsonImporter(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    )
    assert isinstance(params, IVIMParams)


def test_json_save_ideal(capsys, ideal_params, out_json):
    # Test IDEAL
    ideal_params.save_to_json(out_json)
    test_params = IDEALParams(out_json)
    attributes = compare_parameters(ideal_params, test_params)
    compare_attributes(ideal_params, test_params, attributes)
    capsys.readouterr()
    assert True
