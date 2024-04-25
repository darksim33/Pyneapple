import pytest

from pyneapple.fit import parameters
from test_toolbox import ParameterTools


def test_nnls_init_parameters():
    assert parameters.NNLSParams()


def test_nnls_get_basis(nnls_params):
    basis = nnls_params.get_basis()
    assert basis.shape == (
        nnls_params.boundaries.number_points + nnls_params.b_values.shape[0],
        nnls_params.boundaries.number_points,
    )
    assert basis.max() == 1
    assert basis.min() == 0
    assert True


def test_nnls_get_pixel_args(nnls_params, img, seg):
    args = nnls_params.get_pixel_args(img, seg)
    assert args is not None


@pytest.mark.parametrize("seg_number", [1, 2])
def test_nnls_get_seg_args(nnls_params, img, seg, seg_number):
    args = nnls_params.get_seg_args(img, seg, seg_number)
    assert args is not None


def test_nnls_eval_fitting_results(nnls_params):
    assert True


def test_nnls_apply_AUC(nnls_params):
    assert True


def test_nnls_json_save(capsys, nnls_params, out_json):
    # Test NNLS
    nnls_params.save_to_json(out_json)
    test_params = parameters.NNLSParams(out_json)
    attributes = ParameterTools.compare_parameters(nnls_params, test_params)
    ParameterTools.compare_attributes(nnls_params, test_params, attributes)
    capsys.readouterr()
    assert True


# NNLS_CV
def test_nnls_cv_init_parameters():
    assert parameters.NNLSCVParams()


def test_nnlscv_json_save(capsys, nnlscv_params, out_json):
    # Test NNLS CV
    nnlscv_params.save_to_json(out_json)
    test_params = parameters.NNLSCVParams(out_json)
    attributes = ParameterTools.compare_parameters(nnlscv_params, test_params)
    ParameterTools.compare_attributes(nnlscv_params, test_params, attributes)
    capsys.readouterr()
    assert True
