import pytest
from multiprocessing import freeze_support
from functools import wraps
import numpy as np

from pyneapple import FitData
from pyneapple import IVIMSegmentedParams
from pyneapple.fitting.multithreading import multithreader
from pyneapple.fitting.fit import fit_pixel_wise
from pyneapple.fitting.gpubridge import gpu_fitter


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
        "ivim_fit",
        ["ivim_mono_fit_data", "ivim_bi_fit_data", "ivim_tri_fit_data"],
    )
    def test_ivim_pixel_multithreading(self, ivim_fit: FitData, capsys, request):
        ivim_fit_data = request.getfixturevalue(ivim_fit)
        ivim_fit_data.params.n_pools = 4
        ivim_fit_data.fit_pixel_wise(fit_type="multi")
        # capsys.readouterr()
        assert True

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "ivim_fit", ["ivim_mono_fit_data", "ivim_bi_fit_data", "ivim_tri_fit_data"]
    )
    def test_ivim_pixel_sequential(self, ivim_fit: FitData, capsys, request):
        ivim_fit_data = request.getfixturevalue(ivim_fit)
        ivim_fit_data.fit_pixel_wise(fit_type="single")
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

    @pytest.mark.gpu
    def test_biexp_gpu(self, decay_bi, ivim_bi_gpu_params):
        fit_args = decay_bi["fit_array"]
        result = gpu_fitter(
            fit_args,
            ivim_bi_gpu_params,
        )
        assert result is not None

    def test_triexp_gpu(self, decay_tri, ivim_tri_gpu_params):
        fit_args = decay_tri["fit_args"]
        results = gpu_fitter(fit_args, ivim_tri_gpu_params)
        assert results is not None


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
        self, img, seg, ivim_tri_t1_segmented_params_file, out_nii, capsys, options
    ):
        fit_data = FitData(
            img,
            seg,
            ivim_tri_t1_segmented_params_file,
        )
        fit_data.params.set_options(
            options["fixed_component"], options["fixed_t1"], options["reduced_b_values"]
        )
        if not options["fixed_t1"]:
            fit_data.params.mixing_time = None
        fit_data.fit_ivim_segmented(fit_type="multi")
        assert True
        capsys.readouterr()

    def test_ivim_segmented_bi(
        self, img, seg, ivim_bi_segmented_params_file, out_nii, capsys
    ):
        fit_data = FitData(
            img,
            seg,
            ivim_bi_segmented_params_file,
        )
        fit_data.params.set_options(
            fixed_component="D_slow", fixed_t1=False, reduced_b_values=None
        )
        fit_data.params.mixing_time = None
        fit_data.fit_ivim_segmented(fit_type="multi")
        assert True
        capsys.readouterr()
