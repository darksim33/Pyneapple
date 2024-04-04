import pytest
from multiprocessing import freeze_support
from pathlib import Path

from pyneapple.fit import FitData
from pyneapple.utils.nifti import Nii, NiiSeg


@pytest.fixture(scope="module")
def test_IDEAL_ivim():
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii.gz"))
    json = Path(
        Path(__file__).parent.parent,
        "./resources/fitting/default_params_IDEAL_tri.json",
    )
    return FitData("IDEAL", json, img, seg)


def test_IDEAL_ivim_sequential(test_IDEAL_ivim):
    freeze_support()
    test_IDEAL_ivim.fit_IDEAL(multi_threading=False)
    test_IDEAL_ivim.fit_results.save_results_to_excel("test_IDEAL_results.xlsx")
    test_IDEAL_ivim.fit_results.save_fitted_parameters_to_nii(
        "test_IDEAL_results.nii", shape=test_IDEAL_ivim.img.array.shape
    )

    assert True


def test_IDEAL_ivim_multithreading(test_IDEAL_ivim):
    freeze_support()
    test_IDEAL_ivim.fit_IDEAL(multi_threading=True)
    test_IDEAL_ivim.fit_results.save_results_to_excel("test_IDEAL_results.xlsx")
    test_IDEAL_ivim.fit_results.save_fitted_parameters_to_nii(
        "test_IDEAL_results.nii", shape=test_IDEAL_ivim.img.array.shape
    )

    assert True
