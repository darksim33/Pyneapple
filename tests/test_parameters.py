"""Test Basic Parameter Class and Boundaries"""

import pytest
from pathlib import Path
import numpy as np
from pyneapple.parameters.parameters import BaseParams


def test_load_b_values(root):
    parameters = BaseParams()
    file = root / r"tests/.data/test_bvalues.bval"
    assert file.is_file()
    parameters.load_b_values(file)
    b_values = np.array(
        [
            0,
            50,
            100,
            150,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            1000,
            1200,
            1400,
            1600,
            1800,
        ]
    )
    assert b_values.all() == parameters.b_values.all()


def test_get_pixel_args(img, seg):
    parameters = BaseParams()
    args = parameters.get_pixel_args(img, seg)
    assert len(list(args)) == len(np.where(seg != 0)[0])


@pytest.mark.parametrize("seg_number", [1, 2])
def test_get_seg_args_seg_number(img, seg, seg_number):
    parameters = BaseParams()
    args = parameters.get_seg_args(img, seg, seg_number)
    assert len(list(args)) == 1


def test_boundaries_ivim(ivim_bi_params):
    start_values = np.random.randint(2, 100, 4)
    lower_bound = start_values - 1
    upper_bound = start_values + 1
    bounds = {
        "D": {
            "slow": np.array([start_values[1], lower_bound[1], upper_bound[1]]),
            "fast": np.array([start_values[3], lower_bound[3], upper_bound[3]]),
        },
        "f": {
            "slow": np.array([start_values[0], lower_bound[0], upper_bound[0]]),
            "fast": np.array([start_values[2], lower_bound[2], upper_bound[2]]),
        },
    }
    ivim_bi_params.boundaries.dict.update(bounds)
    assert (start_values == ivim_bi_params.boundaries.start_values).all()
    assert (lower_bound == ivim_bi_params.boundaries.lower_bounds).all()
    assert (upper_bound == ivim_bi_params.boundaries.upper_bounds).all()

    start_values = np.random.randint(2, 100, 3)
    lower_bound = start_values - 1
    upper_bound = start_values + 1
    bounds = {
        "D": {
            "slow": np.array([start_values[1], lower_bound[1], upper_bound[1]]),
            "fast": np.array([start_values[2], lower_bound[2], upper_bound[2]]),
        },
        "f": {
            "slow": np.array([start_values[0], lower_bound[0], upper_bound[0]]),
        },
    }
    ivim_bi_params.boundaries.dict.clear()
    ivim_bi_params.boundaries.dict.update(bounds)
    assert (start_values == ivim_bi_params.boundaries.start_values).all()
    assert (lower_bound == ivim_bi_params.boundaries.lower_bounds).all()
    assert (upper_bound == ivim_bi_params.boundaries.upper_bounds).all()
