import pytest
from pathlib import Path
from functools import wraps
from multiprocessing import freeze_support

from pyneapple.utils.nifti import Nii, NiiSeg
from pyneapple.fit import FitData


# Decorators
def freeze_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    freeze_support()
    return wrapper


# Segmented sequential fitting
@pytest.mark.parametrize("reg_order", [0, 1, 2, 3])
def test_nnls_segmented_reg(reg_order, nnls_fit_data: FitData, out_nii: Path, capsys):
    nnls_fit_data.fit_params.reg_order = reg_order
    nnls_fit_data.fit_segmentation_wise()

    nii_dyn = Nii().from_array(nnls_fit_data.fit_results.spectrum)
    nii_dyn.save(out_nii)
    capsys.readouterr()
    assert True


def test_nnls_segmented_reg_cv(nnlscv_fit_data: FitData, out_nii: Path, capsys):
    nnlscv_fit_data.fit_segmentation_wise()

    nii_dyn = Nii().from_array(nnlscv_fit_data.fit_results.spectrum)
    nii_dyn.save(out_nii)
    capsys.readouterr()
    assert True


# Multithreading
@freeze_me
@pytest.mark.slow
@pytest.mark.parametrize("reg_order", [0, 1, 2, 3])
def test_nnls_pixel_multi_reg(reg_order, nnls_fit_data: FitData, capsys):
    nnls_fit_data.fit_params.reg_order = 0
    nnls_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True
    assert True


@freeze_me
@pytest.mark.slow
@pytest.mark.skip("Not working properly atm.")
def test_nnls_pixel_multi_reg_cv(nnlscv_fit_data: FitData, capsys):
    nnlscv_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True
