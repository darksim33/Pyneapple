import pytest
import random
import numpy as np
from scipy import signal

from pyneapple.fit import parameters
from pyneapple.utils import NiiSeg

from test_toolbox import ParameterTools


class TestNNLSParameters:
    @pytest.fixture
    def nii_seg_reduced(self):
        array = np.ones((2, 2, 2, 1))
        nii = NiiSeg().from_array(array)
        return nii

    @pytest.fixture
    def nnls_fit_results(self, nnls_params):
        # Get D Values from bins
        bins = nnls_params.get_bins()
        d_value_indexes = random.sample(
            np.linspace(0, len(bins) - 1, num=len(bins)).astype(int).tolist(), 3
        )
        self.d_values = np.array([bins[i] for i in d_value_indexes])

        # Get f Values
        f1 = random.uniform(0, 1)
        f2 = random.uniform(0, 1)
        while f1 + f2 >= 1:
            f1 = random.uniform(0, 1)
            f2 = random.uniform(0, 1)
        f3 = 1 - f1 - f2
        self.f_values = np.array([f1, f2, f3])

        # Get Spectrum
        spectrum = np.zeros(nnls_params.boundaries.number_points)
        for idx, d in enumerate(d_value_indexes):
            spectrum = spectrum + self.f_values[idx] * signal.unit_impulse(
                nnls_params.boundaries.number_points,
                d_value_indexes[idx],
            )

        self.pixel_indexes = [(0, 0, 0)]
        results = list()
        for idx in self.pixel_indexes:
            results.append((idx, spectrum))

        return results

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

    @pytest.mark.parametrize("seg_number", [1, 2])
    def test_nnls_get_seg_args(self, nnls_params, img, seg, seg_number):
        args = nnls_params.get_seg_args(img, seg, seg_number)
        assert args is not None

    def test_nnls_eval_fitting_results(
        self, nnls_fit_results, nnls_params, nii_seg_reduced
    ):
        results = nnls_params.eval_fitting_results(nnls_fit_results, nii_seg_reduced)
        self.results = results
        for idx in self.pixel_indexes:
            assert results.f[idx].all() == self.f_values.all()
            assert results.d[idx].all() == self.d_values.all()

    def test_nnls_apply_auc(self, nnls_params):
        results = nnls_params.apply_AUC_to_results(self.results)
        assert results

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
