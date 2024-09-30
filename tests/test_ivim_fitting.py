import pytest
from multiprocessing import freeze_support
from functools import wraps
import numpy as np

from pyneapple.fit import FitData
from pyneapple.fit.parameters import IVIMSegmentedParams
from pyneapple.utils.multithreading import multithreader


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


class TestIVIMSegmentedFitting:
    @pytest.mark.slow
    def test_ivim_segmented_first_fit(
        self, img, seg, ivim_tri_params_file, ivim_mono_params
    ):
        pixel_args_mono = ivim_mono_params.get_pixel_args(img, seg)
        results_mono = dict(
            multithreader(ivim_mono_params.fit_function, pixel_args_mono, None)
        )

        ivim_tri_segmented_params = IVIMSegmentedParams(ivim_tri_params_file)
        ivim_tri_segmented_params.set_options(
            fixed_component="D_slow", fixed_t1=False, reduced_b_values=None
        )

        pixel_args_segmented = ivim_tri_segmented_params.get_pixel_args_fixed(img, seg)
        results_segmented = dict(
            multithreader(
                ivim_tri_segmented_params.params_fixed.fit_function,
                pixel_args_segmented,
                None,
            )
        )
        for key in results_mono:
            assert results_mono[key].all() == results_segmented[key].all()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "options",
        [
            {"fixed_component": "D_slow", "fixed_t1": False, "reduced_b_values": None},
            # {"fixed_component": "D_slow", "fixed_t1": True, "reduced_b_values": None},
            {
                "fixed_component": "D_slow",
                "fixed_t1": False,
                "reduced_b_values": [100, 150, 200, 250, 350, 450, 550, 650, 750],
            },
            # {
            #     "fixed_component": "D_slow",
            #     "fixed_t1": True,
            #     "reduced_b_values": [100, 150, 200, 250, 350, 450, 550, 650, 750],
            # },
        ],
    )
    def test_ivim_segmented_tri(
        self, img, seg, ivim_tri_t1_params_file, out_nii, capsys, options
    ):
        fit_data = FitData("IVIMSegmented", ivim_tri_t1_params_file, img, seg)
        fit_data.params.set_options(
            options["fixed_component"], options["fixed_t1"], options["reduced_b_values"]
        )
        if not options["fixed_t1"]:
            fit_data.params.TM = None
        fit_data.fit_ivim_segmented(False)
        assert True
        capsys.readouterr()

    def test_ivim_segmented_bi(self, img, seg, ivim_bi_params_file, out_nii, capsys):
        fit_data = FitData("IVIMSegmented", ivim_bi_params_file, img, seg)
        fit_data.params.set_options(
            fixed_component="D_slow", fixed_t1=False, reduced_b_values=None
        )
        fit_data.params.TM = None
        fit_data.fit_ivim_segmented(False)
        assert True
        capsys.readouterr()
