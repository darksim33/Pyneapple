import pytest
from multiprocessing import freeze_support
from pathlib import Path

from src.fit.fit import FitData
from src.utils import Nii, NiiSeg


@pytest.fixture(scope="module")
def test_ideal_ivim():
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii.gz"))
    json = Path(
        Path(__file__).parent.parent,
        "./resources/fitting/default_params_IDEAL_tri.json",
    )
    return FitData("IDEAL", json, img, seg)


def test_ideal_ivim_sequential(test_ideal_ivim):
    freeze_support()
    test_ideal_ivim.fit_ideal(multi_threading=False)
    test_ideal_ivim.fit_results.save_results_to_excel("test_ideal_results.xlsx")
    test_ideal_ivim.fit_results.save_fitted_parameters_to_nii(
        "test_ideal_results.nii", shape=test_ideal_ivim.img.array.shape
    )

    assert True


def test_ideal_ivim_multithreading(test_ideal_ivim):
    freeze_support()
    test_ideal_ivim.fit_ideal(multi_threading=True)
    test_ideal_ivim.fit_results.save_results_to_excel("test_ideal_results.xlsx")
    test_ideal_ivim.fit_results.save_fitted_parameters_to_nii(
        "test_ideal_results.nii", shape=test_ideal_ivim.img.array.shape
    )

    assert True
