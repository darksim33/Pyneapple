import pytest
from pathlib import Path

from pyneapple.fit import parameters, FitData
from pyneapple.utils.nifti import Nii, NiiSeg


@pytest.fixture
def img():
    file = Path(r"../data/test_img.nii")
    if file.exists():
        assert True
    else:
        assert False
    return Nii(file)


@pytest.fixture
def seg():
    file = Path(r"../data/test_mask.nii")
    if file.exists():
        assert True
    else:
        assert file.exists()
    return NiiSeg(file)


@pytest.fixture
def out_json():
    file = Path(r".out/test_params.json")
    return file


@pytest.fixture
def out_nii():
    file = Path(r".out/out_nii.nii.gz")
    return file


@pytest.fixture
def out_excel():
    file = Path(r".out/out_excel.xlsx")
    return file


# IVIM
@pytest.fixture
def ivim_mono_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_mono.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IVIMParams(file)


@pytest.fixture
def ivim_bi_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_bi.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IVIMParams(file)


@pytest.fixture
def ivim_tri_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IVIMParams(file)


@pytest.fixture
def ivim_mono_fit_data(img, seg, ivim_mono_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.fit_params = ivim_mono_params
    return fit_data


@pytest.fixture
def ivim_bi_fit_data(img, seg, ivim_bi_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.fit_params = ivim_bi_params
    return fit_data


@pytest.fixture
def ivim_tri_fit_data(img, seg, ivim_tri_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.fit_params = ivim_tri_params
    return fit_data


# NNLS
@pytest.fixture
def nnls_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.NNLSParams(file)


@pytest.fixture
def nnlscv_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.NNLSCVParams(file)


@pytest.fixture
def nnls_fit_data(img, seg, nnls_params):
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
def nnlscv_fit_data(img, seg, nnlscv_params):
    fit_data = FitData(
        "NNLSCV",
        None,
        img,
        seg,
    )
    fit_data.fit_params = nnlscv_params
    fit_data.fit_params.max_iter = 10000
    return fit_data


# IDEAL
@pytest.fixture
def ideal_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IDEAL_bi.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IDEALParams(file)


@pytest.fixture
def test_ideal_fit_data(img, seg, ideal_params):
    fit_data = FitData(
        "IDEAL",
        None,
        img,
        seg,
    )
    fit_data.fit_params = ideal_params
    return fit_data
