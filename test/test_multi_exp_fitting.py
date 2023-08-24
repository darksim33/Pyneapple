import pytest
import numpy as np
from multiprocessing import freeze_support
from pathlib import Path

from src.utils import Nii, NiiSeg
from src.fit import fit
from src.fit.parameters import MultiTest


@pytest.fixture
def mono_exp():
    img = Nii(Path(r"../data/kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    fitData = fit.FitData("MonoExp", img, seg)
    fitData.fit_params = MultiTest()
    fitData.fit_params.boundaries.x0 = np.array(
        [
            0.1,  # D_fast
            210,  # S_0
        ]
    )
    fitData.fit_params.boundaries.lb = np.array(
        [
            0.01,  # D_fast
            10,  # S_0
        ]
    )
    fitData.fit_params.boundaries.ub = np.array(
        [
            0.5,  # D_fast
            1000,  # S_0
        ]
    )
    return fitData


@pytest.fixture
def bi_exp():
    img = Nii(Path(r"../data/kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    fitData = fit.FitData("BiExp", img, seg)
    fitData.fit_params = MultiTest()
    fitData.fit_params.boundaries.x0 = np.array(
        [
            0.1,  # D_fast
            0.005,  # D_inter
            0.1,  # f_fast
            210,  # S_0
        ]
    )
    fitData.fit_params.boundaries.lb = np.array(
        [
            0.01,  # D_fast
            0.003,  # D_inter
            0.01,  # f_fast
            10,  # S_0
        ]
    )
    fitData.fit_params.boundaries.ub = np.array(
        [
            0.5,  # D_fast
            0.01,  # D_inter
            0.7,  # f_fast
            1000,  # S_0
        ]
    )
    return fitData


@pytest.fixture
def tri_exp():
    img = Nii(Path(r"../data/kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    fitData = fit.FitData("TriExp", img, seg)
    fitData.fit_params = MultiTest()
    fitData.fit_params.boundaries.x0 = np.array(
        [
            0.1,  # D_fast
            0.005,  # D_inter
            0.0015,  # D_slow
            0.1,  # f_fast
            0.2,  # f_inter
            210,  # S_0
        ]
    )
    fitData.fit_params.boundaries.lb = np.array(
        [
            0.01,  # D_fast
            0.003,  # D_inter
            0.0011,  # D_slow
            0.01,  # f_fast
            0.1,  # f_inter
            10,  # S_0
        ]
    )
    fitData.fit_params.boundaries.ub = np.array(
        [
            0.5,  # D_fast
            0.01,  # D_inter
            0.003,  # D_slow
            0.7,  # f_fast
            0.7,  # f_inter
            1000,  # S_0
        ]
    )
    return fitData


def test_mono_exp_pixel_sequential(mono_exp):
    mono_exp.fit_pixel_wise(multi_threading=False)
    assert True


def test_bi_exp_pixel_sequential(bi_exp):
    bi_exp.fit_pixel_wise(multi_threading=False)
    assert True


def test_tri_exp_pixel_sequential(tri_exp):
    tri_exp.fit_pixel_wise(multi_threading=False)
    assert True
