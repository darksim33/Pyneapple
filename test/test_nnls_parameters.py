import pytest

from pyneapple.fit import parameters
from pyneapple.fit import Results
from pyneapple.utils import NiiSeg

from test_toolbox import ParameterTools


class TestNNLSParameters:
    def test_nnls_init_parameters(self):
        assert parameters.NNLSParams()

    def test_nnls_get_basis(self, nnls_params):
        basis = nnls_params.get_basis()
        assert basis.shape == (
            nnls_params.boundaries.number_points + nnls_params.b_values.shape[0],
            nnls_params.boundaries.number_points,
        )
        assert basis.max() == 1
        assert basis.min() == 0
        assert True

    def test_nnls_get_pixel_args(self, nnls_params, img, seg):
        args = nnls_params.get_pixel_args(img, seg)
        assert args is not None

    @pytest.mark.parametrize("seg_number", [1, 2, 3])
    def test_nnls_get_seg_args(self, nnls_params, img, seg, seg_number):
        args = nnls_params.get_seg_args(img, seg, seg_number)
        assert args is not None

    def test_nnls_eval_fitting_results(
        self, nnls_fit_results, nnls_params, nii_seg_reduced
    ):
        results = nnls_params.eval_fitting_results(nnls_fit_results[0])
        fit_results = Results()
        fit_results.update_results(results)
        for idx in nnls_fit_results[3]:
            assert fit_results.f[idx].all() == nnls_fit_results[2][idx].all()
            assert fit_results.d[idx].all() == nnls_fit_results[1][idx].all()

    # def test_nnls_spectrum_dict(self, nnls_fit_results, nnls_params, nii_seg_reduced):
    #     results = nnls_params.eval_fitting_results(nnls_fit_results, nii_seg_reduced)
    #     assert True

    @pytest.mark.order(after="test_nnls_eval_fitting_results")
    def test_nnls_apply_auc(self, nnls_params, nnls_fit_results, nii_seg_reduced):
        results = nnls_params.eval_fitting_results(nnls_fit_results[0])
        fit_results = Results()
        fit_results.update_results(results)
        assert nnls_params.apply_AUC_to_results(fit_results)

    def test_nnls_json_save(self, capsys, nnls_params, out_json):
        # Test NNLS
        nnls_params.save_to_json(out_json)
        test_params = parameters.NNLSParams(out_json)
        attributes = ParameterTools.compare_parameters(nnls_params, test_params)
        ParameterTools.compare_attributes(nnls_params, test_params, attributes)
        capsys.readouterr()
        assert True

    # NNLS_CV
    def test_nnls_cv_init_parameters(self):
        assert parameters.NNLSCVParams()

    def test_nnlscv_json_save(self, capsys, nnlscv_params, out_json):
        # Test NNLS CV
        nnlscv_params.save_to_json(out_json)
        test_params = parameters.NNLSCVParams(out_json)
        attributes = ParameterTools.compare_parameters(nnlscv_params, test_params)
        ParameterTools.compare_attributes(nnlscv_params, test_params, attributes)
        capsys.readouterr()
        assert True
