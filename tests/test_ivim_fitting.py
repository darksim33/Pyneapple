import pytest
from multiprocessing import freeze_support
from functools import wraps
import numpy as np

from pyneapple.fit import FitData


# Decorators
def freeze_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    freeze_support()
    return wrapper


# Tests
# @pytest.mark.order(
#     after="test_ivim_parameters.py::TestIVIMParameters::test_ivim_json_save"
# )
class TestIVIMFitting:
    def test_ivim_tri_segmented(self, ivim_tri_fit_data: FitData, capsys):
        ivim_tri_fit_data.fit_segmentation_wise()
        capsys.readouterr()
        assert True

    @freeze_me
    @pytest.mark.parametrize(
        "ivim_fit", ["ivim_mono_fit_data", "ivim_bi_fit_data", "ivim_tri_fit_data"]
    )
    def test_ivim_pixel_multithreading(self, ivim_fit: FitData, capsys, request):
        ivim_fit_data = request.getfixturevalue(ivim_fit)
        ivim_fit_data.params.n_pools = 4
        ivim_fit_data.fit_pixel_wise(multi_threading=True)
        capsys.readouterr()
        assert True

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "ivim_fit", ["ivim_mono_fit_data", "ivim_bi_fit_data", "ivim_tri_fit_data"]
    )
    def test_ivim_pixel_sequential(self, ivim_fit: FitData, capsys, request):
        ivim_fit_data = request.getfixturevalue(ivim_fit)
        ivim_fit_data.fit_pixel_wise(multi_threading=False)
        if not hasattr(self, "fit_data"):
            self.fit_data = {}
        self.fit_data[ivim_fit] = ivim_fit_data
        capsys.readouterr()
        assert True

    def test_ivim_mono_result_to_fit_curve(self, ivim_mono_fit_data: FitData, capsys):
        ivim_mono_fit_data.results.raw[0, 0, 0] = np.array([0.15, 150])
        ivim_mono_fit_data.params.fit_model(
            ivim_mono_fit_data.params.b_values,
            *ivim_mono_fit_data.results.raw[0, 0, 0].tolist()
        )
        capsys.readouterr()
        assert True

    @pytest.mark.order(after="test_ivim_pixel_sequential")
    def test_ivim_tri_result_to_nii(self, ivim_tri_fit_data: FitData, out_nii, capsys):
        if not ivim_tri_fit_data.results.d:
            if hasattr(self, "fit_data") and self.fit_data["ivim_try_fit_data"]:
                ivim_tri_fit_data = self.fit_data["ivim_try_fit_data"]
            else:
                ivim_tri_fit_data.fit_pixel_wise()

        ivim_tri_fit_data.results.save_fitted_parameters_to_nii(
            out_nii,
            ivim_tri_fit_data.img.array.shape,
            parameter_names=ivim_tri_fit_data.params.boundaries.parameter_names,
        )
        capsys.readouterr()
        assert True

    @pytest.mark.slow
    def test_ivim_segmented_tri(
        self, img, seg, ivim_tri_t1_params_file, out_nii, capsys
    ):
        fit_data = FitData("IVIMSegmented", ivim_tri_t1_params_file, img, seg)
        fit_data.params.set_options(
            fixed_component="D_slow", fixed_t1=True, reduced_b_values=None
        )
        fit_data.fit_ivim_segmented(False)
        assert True
        capsys.readouterr()
