import pytest
from multiprocessing import freeze_support
from functools import wraps
from pathlib import Path
import numpy as np

from pyneapple.fit import FitData, parameters


# Decorators
def freeze_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    freeze_support()
    return wrapper


# Tests
def test_ivim_tri_segmented(ivim_tri_fit_data: FitData, capsys):
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


def test_ivim_mono_pixel_sequential(ivim_mono_fit_data: FitData, capsys):
    ivim_mono_fit_data.fit_pixel_wise(multi_threading=False)
    capsys.readouterr()
    assert True


def test_ivim_bi_pixel_sequential(ivim_bi_fit_data: FitData, capsys):
    ivim_bi_fit_data.fit_pixel_wise(multi_threading=False)
    capsys.readouterr()
    assert True


def test_ivim_tri_pixel_sequential(ivim_tri_fit_data: FitData, capsys):
    ivim_tri_fit_data.fit_pixel_wise(multi_threading=False)
    capsys.readouterr()
    assert True
    return ivim_tri_fit_data


def test_ivim_mono_result_to_fit_curve(ivim_mono_fit_data: FitData, capsys):
    ivim_mono_fit_data.fit_results.raw[0, 0, 0] = np.array([0.15, 150])
    ivim_mono_fit_data.fit_params.fit_model(
        ivim_mono_fit_data.fit_params.b_values,
        *ivim_mono_fit_data.fit_results.raw[0, 0, 0].tolist()
    )
    capsys.readouterr()
    assert True


def test_ivim_tri_result_to_nii(ivim_tri_fit_data: FitData, out_nii, capsys):
    if not ivim_tri_fit_data.fit_results.d:
        ivim_tri_fit_data = test_ivim_tri_pixel_sequential(ivim_tri_fit_data, capsys)

    ivim_tri_fit_data.fit_results.save_fitted_parameters_to_nii(
        out_nii,
        ivim_tri_fit_data.img.array.shape,
        parameter_names=ivim_tri_fit_data.fit_params.boundaries.parameter_names,
    )
    capsys.readouterr()
    assert True
