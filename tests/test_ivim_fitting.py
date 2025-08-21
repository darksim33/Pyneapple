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
    def test_ivim_tri_segmented(self, ivim_tri_fit_data: FitData):
        ivim_tri_fit_data.fit_segmentation_wise()
        assert True

    @freeze_me
    @pytest.mark.parametrize(
        "ivim_fit",
        ["ivim_mono_fit_data", "ivim_bi_fit_data", "ivim_tri_fit_data"],
    )
    def test_ivim_pixel_multithreading(self, ivim_fit: FitData, request):
        ivim_fit_data = request.getfixturevalue(ivim_fit)
        ivim_fit_data.params.n_pools = 4
        ivim_fit_data.fit_pixel_wise(fit_type="multi")
        assert True

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "ivim_fit", ["ivim_mono_fit_data", "ivim_bi_fit_data", "ivim_tri_fit_data"]
    )
    def test_ivim_pixel_sequential(self, ivim_fit: FitData, request):
        ivim_fit_data = request.getfixturevalue(ivim_fit)
        ivim_fit_data.fit_pixel_wise(fit_type="single")
        if not hasattr(self, "fit_data"):
            self.fit_data = {}
        self.fit_data[ivim_fit] = ivim_fit_data
        assert True

    def test_ivim_mono_result_to_fit_curve(self, ivim_mono_fit_data: FitData):
        ivim_mono_fit_data.results.raw[0, 0, 0] = np.array([0.15, 150])
        ivim_mono_fit_data.params.fit_model.model(
            ivim_mono_fit_data.params.b_values,
            *ivim_mono_fit_data.results.raw[0, 0, 0].tolist(),
        )
        assert True

    @pytest.mark.gpu
    def test_biexp_gpu(self, decay_bi, ivim_bi_gpu_params):
        fit_args = decay_bi["fit_args"]
        result = gpu_fitter(
            fit_args,
            ivim_bi_gpu_params,
        )
        assert result is not None

    @pytest.mark.gpu
    def test_triexp_gpu(self, decay_tri, ivim_tri_gpu_params):
        fit_args = decay_tri["fit_args"]
        results = gpu_fitter(fit_args, ivim_tri_gpu_params)
        assert results is not None


class TestIVIMSegmentedFitting:
    def test_ivim_segmented_first_fit(
        self, img, seg, ivim_tri_t1_params_file, ivim_mono_params
    ):
        """
        Test that the first fit of IVIM segmented fitting produces results
        consistent with mono-exponential fitting.

        This test validates that the initial segmented fit step matches
        the mono-exponential approach within acceptable tolerance.
        """
        # Perform mono-exponential fitting for baseline comparison
        mono_results = self._perform_mono_fitting(img, seg, ivim_mono_params)

        # Set up and perform segmented fitting (first fit only)
        segmented_results = self._perform_segmented_first_fit(
            img, seg, ivim_tri_t1_params_file
        )

        # Compare results between mono and segmented first fit
        self._assert_results_match(mono_results, segmented_results)

    def _perform_mono_fitting(self, img, seg, ivim_mono_params):
        """Perform mono-exponential fitting and return results."""
        pixel_args_mono = ivim_mono_params.get_pixel_args(img, seg)
        return multithreader(ivim_mono_params.fit_function, pixel_args_mono, None)

    def _perform_segmented_first_fit(self, img, seg, ivim_tri_t1_params_file):
        """Set up and perform the first fit of segmented IVIM fitting."""
        # Initialize segmented parameters
        segmented_params = IVIMSegmentedParams(ivim_tri_t1_params_file)

        # Configure segmented fitting parameters
        self._configure_segmented_params(segmented_params)

        # Set up the parameters
        segmented_params.set_up()

        # Perform the first segmented fit
        pixel_args_segmented = segmented_params.get_pixel_args_fit1(img, seg)
        return multithreader(
            segmented_params.params_1.fit_function,
            pixel_args_segmented,
            None,
        )

    def _configure_segmented_params(self, segmented_params):
        """Configure the segmented parameters with test-specific settings."""
        segmented_params.fixed_component = "D_1"
        segmented_params.fit_model.mixing_time = 20
        segmented_params.fixed_t1 = False
        segmented_params.fit_model.fit_t1 = False
        segmented_params.reduced_b_values = None

    def _assert_results_match(self, mono_results, segmented_results, rtol=1e-5):
        """Assert that mono and segmented results match within tolerance."""
        assert len(mono_results) == len(segmented_results), (
            f"Result count mismatch: mono={len(mono_results)}, "
            f"segmented={len(segmented_results)}"
        )

        for idx, (mono_result, segmented_result) in enumerate(
            zip(mono_results, segmented_results)
        ):
            try:
                np.testing.assert_allclose(
                    mono_result[1],
                    segmented_result[1],
                    rtol=rtol,
                    err_msg=f"Results differ at index {idx}",
                )
            except AssertionError as e:
                pytest.fail(f"Fitting results comparison failed at pixel {idx}: {e}")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "options",
        [
            {"fixed_component": "D_1", "fixed_t1": False, "reduced_b_values": None},
            # {"fixed_component": "D_1", "fixed_t1": True, "reduced_b_values": None},
            {
                "fixed_component": "D_1",
                "fixed_t1": False,
                "reduced_b_values": [100, 150, 200, 250, 350, 450, 550, 650, 750],
            },
            # {
            #     "fixed_component": "D_1",
            #     "fixed_t1": True,
            #     "reduced_b_values": [100, 150, 200, 250, 350, 450, 550, 650, 750],
            # },
        ],
    )
    def test_ivim_segmented_tri(
        self, img, seg, ivim_tri_t1_segmented_params_file, out_nii, options
    ):
        fit_data = FitData(
            img,
            seg,
            ivim_tri_t1_segmented_params_file,
        )

        fit_data.params.fixed_component = options["fixed_component"]
        fit_data.params.fixed_t1 = options["fixed_t1"]
        fit_data.params.reduced_b_values = options["reduced_b_values"]
        fit_data.params.set_up()

        fit_data.fit_ivim_segmented(fit_type="single")
        assert True

    def test_ivim_segmented_bi(self, img, seg, ivim_bi_segmented_params_file, out_nii):
        fit_data = FitData(
            img,
            seg,
            ivim_bi_segmented_params_file,
        )
        fit_data.params.fixed_component = "D_1"
        fit_data.params.fixed_t1 = False
        fit_data.params.reduced_b_values = None
        fit_data.params.mixing_time = None
        fit_data.fit_ivim_segmented(fit_type="multi")
        assert True
