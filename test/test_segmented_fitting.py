import pytest
import numpy as np

from pathlib import Path

from src.pyneapple.fit.parameters import IVIMParams
from pyneapple.utils.nifti import Nii, NiiSeg
from pyneapple.fit import FitData


@pytest.fixture
def nnls_fit_data():
    img = Nii(Path(r"../data/test_img.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii"))

    fit_data = FitData(
        "NNLS",
        Path("../src/pyneapple/resources/fitting/default_params_NNLS.json"),
        img,
        seg,
    )
    fit_data.fit_params.max_iter = 10000

    return fit_data


@pytest.fixture
def nnls_cv_fit_data():
    img = Nii(Path(r"../data/test_img.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii"))

    fit_data = FitData(
        "NNLSCV",
        Path("../src/pyneapple/resources/fitting/default_params_NNLSCV.json"),
        img,
        seg,
    )
    fit_data.fit_params.max_iter = 10000

    return fit_data


@pytest.fixture
def tri_exp():
    img = Nii(Path(r"../data/test_img.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii"))
    fit_data = FitData(
        "IVIM",
        Path("../src/pyneapple/resources/fitting/default_params_IVIM_tri.json"),
        img,
        seg,
    )
    return fit_data


def test_nnls_segmented_reg_0(nnls_cv_fit_data: FitData, capsys):
    nnls_cv_fit_data.fit_params.reg_order = 0
    nnls_cv_fit_data.fit_segmentation_wise()

    nii_dyn = Nii().from_array(nnls_cv_fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_seg.nii")
    capsys.readouterr()
    assert True


def test_nnls_segmented_reg_1(nnls_cv_fit_data: FitData, capsys):
    nnls_cv_fit_data.fit_params.reg_order = 1
    nnls_cv_fit_data.fit_segmentation_wise()

    nii_dyn = Nii().from_array(nnls_cv_fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_seg.nii")
    capsys.readouterr()
    assert True


def test_nnls_segmented_reg_2(nnls_cv_fit_data: FitData, capsys):
    nnls_cv_fit_data.fit_params.reg_order = 1
    nnls_cv_fit_data.fit_segmentation_wise()

    nii_dyn = Nii().from_array(nnls_cv_fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_seg.nii")
    capsys.readouterr()
    assert True


def test_nnls_segmented_reg_cv(nnls_cv_fit_data: FitData, capsys):
    nnls_cv_fit_data.fit_segmentation_wise()

    nii_dyn = Nii().from_array(nnls_cv_fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_seg.nii")
    capsys.readouterr()
    assert True


def test_tri_exp_segmented(tri_exp: FitData, capsys):
    tri_exp.fit_segmentation_wise()
    capsys.readouterr()
    assert True
