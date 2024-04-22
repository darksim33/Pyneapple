import pytest
from pathlib import Path

from pyneapple.fit import parameters, FitData
from pyneapple.utils.nifti import Nii, NiiSeg


img = Nii(Path(r"../data/test_img.nii"))
seg = NiiSeg(Path(r"../data/test_mask.nii"))


def capsys_decorator(func, capsys):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    capsys.readouterr()
    return wrapper


@pytest.fixture
def ivim_mono_params():
    return parameters.IVIMParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_mono.json")
    )


@pytest.fixture
def ivim_bi_params():
    return parameters.IVIMParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_bi.json")
    )


@pytest.fixture
def ivim_tri_params():
    return parameters.IVIMParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    )


@pytest.fixture
def nnls_params():
    return parameters.NNLSParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json")
    )


@pytest.fixture
def nnlscv_params():
    return parameters.NNLSCVParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json")
    )


@pytest.fixture
def ideal_params():
    return parameters.IDEALParams(
        Path(r"../src/pyneapple/resources/fitting/default_params_IDEAL_bi.json")
    )


@pytest.fixture
def ivim_mono_fit_data(ivim_mono_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.fit_params = ivim_mono_params
    return fit_data


@pytest.fixture
def ivim_bi_fit_data(ivim_bi_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.fit_params = ivim_bi_params
    return fit_data


@pytest.fixture
def ivim_tri_fit_data(ivim_tri_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.fit_params = ivim_tri_params
    return fit_data


@pytest.fixture
def nnls_fit_data(nnls_params):
    fit_data = FitData(
        "NNLS",
        None,
        img,
        seg,
    )
    fit_data.fit_params = nnls_params
    fit_data.fit_params.max_iter = 10000
    return fit_data


@pytest.fixture
def nnlscv_fit_data(nnlscv_params):
    fit_data = FitData(
        "NNLSCV",
        None,
        img,
        seg,
    )
    fit_data.fit_params = nnlscv_params
    fit_data.fit_params.max_iter = 10000
    return fit_data
