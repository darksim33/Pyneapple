import pytest
from multiprocessing import freeze_support
from pathlib import Path

from src.fit.ideal import fit_ideal
from src.fit.parameters import IDEALParams
from src.utils import Nii, NiiSeg


@pytest.fixture(scope="module")
def test_ideal_ivim():
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii.gz"))
    json = Path(
        Path(__file__).parent.parent,
        "./resources/fitting/default_params_ideal.json",
    )
    params = IDEALParams(json)
    return [img, seg, params]


def test_ideal_ivim_sequential(test_ideal_ivim):
    freeze_support()
    img, seg, params = test_ideal_ivim
    fit = fit_ideal(img, seg, params, debug=False, multithreading=False)
    fit_results = params.eval_fitting_results(fit, seg)
    fit_results.save_results_to_excel("test_ideal_results.xlsx")
    fit_results.save_results_to_nii("test_ideal_results.nii", img_dim=img.array.shape)

    assert True


def test_ideal_ivim_multithreading(test_ideal_ivim):
    freeze_support()
    img, seg, params = test_ideal_ivim
    fit = fit_ideal(img, seg, params, debug=False, multithreading=True)
    fit_results = params.eval_fitting_results(fit, seg)
    fit_results.save_results_to_excel("test_ideal_results.xlsx")
    fit_results.save_results_to_nii("test_ideal_results.nii", img_dim=img.array.shape)

    assert True
