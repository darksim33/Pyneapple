import pytest
from multiprocessing import freeze_support
from functools import wraps
from pathlib import Path

from pyneapple.fit import FitData, parameters


# Decorators
def freeze_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    freeze_support()
    return wrapper


# Tests
def test_tri_exp_segmented(ivim_tri_fit_data: FitData, capsys):
    ivim_tri_fit_data.fit_segmentation_wise()
    capsys.readouterr()
    assert True


@freeze_me
def test_ivim_mono_pixel_multithreading(ivim_mono_fit_data: FitData, capsys):
    ivim_mono_fit_data.fit_params.n_pools = 4
    ivim_mono_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True


@freeze_me
def test_ivim_bi_pixel_multithreading(ivim_bi_fit_data: FitData, capsys):
    ivim_bi_fit_data.fit_params.n_pools = 4
    ivim_bi_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True


@freeze_me
def test_ivim_tri_pixel_multithreading(ivim_tri_fit_data: FitData, capsys):
    ivim_tri_fit_data.fit_params.n_pools = 4
    ivim_tri_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True
