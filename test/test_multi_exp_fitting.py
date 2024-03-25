import pytest
import numpy as np
from pathlib import Path

from pyneapple.utils.nifti import Nii, NiiSeg
from pyneapple.fit import fit


@pytest.fixture
def mono_exp():
    img = Nii(Path(r"kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    fit_data = fit.FitData(
        "IVIM", r"../resources/fitting/default_params_IVIM_mono.json", img, seg
    )
    return fit_data


@pytest.fixture
def bi_exp():
    img = Nii(Path(r"kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    fit_data = fit.FitData(
        "IVIM", r"../resources/fitting/default_params_IVIM_bi.json", img, seg
    )
    return fit_data


@pytest.fixture
def tri_exp():
    img = Nii(Path(r"kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    fit_data = fit.FitData(
        "IVIM", r"../resources/fitting/default_params_IVIM_tri.json", img, seg
    )
    return fit_data


def test_mono_exp_pixel_sequential(mono_exp: fit.FitData):
    mono_exp.fit_pixel_wise(multi_threading=False)
    assert True


def test_bi_exp_pixel_sequential(bi_exp: fit.FitData):
    bi_exp.fit_pixel_wise(multi_threading=False)
    assert True


def test_tri_exp_pixel_sequential(tri_exp: fit.FitData):
    tri_exp.fit_pixel_wise(multi_threading=False)
    assert True
    return tri_exp


def test_mono_exp_result_to_fit_curve(mono_exp: fit.FitData):
    mono_exp.fit_results.raw[0, 0, 0] = np.array([0.15, 150])  #
    mono_exp.fit_params.fit_model(
        mono_exp.fit_params.b_values, *mono_exp.fit_results.raw[0, 0, 0].tolist()
    )
    assert True


def test_tri_exp_result_to_nii(tri_exp: fit.FitData):
    if not tri_exp.fit_results.d:
        tri_exp = test_tri_exp_pixel_sequential(tri_exp)

    tri_exp.fit_results.save_fitted_parameters_to_nii(
        r"test_ivim_pixel_fit.nii", tri_exp.img.array.shape
    )
    assert True
