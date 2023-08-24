import pytest
from pathlib import Path
from multiprocessing import freeze_support

from src.utils import Nii, NiiSeg
from src.fit import fit


@pytest.fixture
def fit_data():
    freeze_support()
    img = Nii(Path(r"../data/01_img.nii"))
    seg = NiiSeg(Path(r"../data/01_prostate.nii.gz"))

    fit_data = fit.FitData("NNLS", img, seg)
    fit_data.fit_params.max_iter = 10000

    return fit_data


def test_nnls_pixel_sequential_reg_0(fit_data):
    fit_data.fit_params.reg_order = 0
    fit_data.fit_pixel_wise(multi_threading=False)

    nii_dyn = Nii().from_array(fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_pixel_seq_reg_0.nii")
    assert True


def test_nnls_pixel_sequential_reg_1(fit_data):
    fit_data.fit_params.reg_order = 1
    fit_data.fit_pixel_wise(multi_threading=False)

    nii_dyn = Nii().from_array(fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_pixel_seq_reg_1.nii")
    assert True


def test_nnls_pixel_sequential_reg_2(fit_data):
    fit_data.fit_params.reg_order = 2
    fit_data.fit_pixel_wise(multi_threading=False)

    nii_dyn = Nii().from_array(fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_pixel_seq_reg_2.nii")
    assert True
