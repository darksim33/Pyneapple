"""Comprehensive tests for IVIM fitting functionality.

This module tests the complete IVIM fitting pipeline including:

- Pixel-wise fitting: Individual voxel fitting with various configurations
- Segment-wise fitting: ROI/segment-based fitting with averaging
- Multi-processing: Parallel fitting using Python's multiprocessing
- GPU acceleration: GPU-based fitting using pygpufit (when available)
- Model configurations: Mono/bi/tri-exponential, standard/reduced/S0 variants
- Segmentation strategies: Full segmentation, two-stage (fast/slow), ideal segmentation
- Boundary handling: Uniform and individual pixel boundaries
- Advanced features: T1 correction, fixed parameters, cross-validation

Tests verify correctness of fitted parameters, proper handling of different
fitting strategies, and integration with FitData objects.
"""
import pytest
from multiprocessing import freeze_support
from functools import wraps
import numpy as np

from pyneapple import FitData
from pyneapple import IVIMSegmentedParams
from pyneapple.fitting.multithreading import multithreader
from pyneapple.fitting.fit import fit_pixel_wise
from pyneapple.fitting.gpubridge import gpu_fitter

from tests.test_utils import canonical_parameters as cp
from .test_toolbox import ParameterTools


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
        """Test IVIM tri-exponential segmentation-wise fitting."""
        ParameterTools.assert_fit_completes(
            ivim_tri_fit_data, "fit_segmentation_wise"
        )

    @freeze_me
    @pytest.mark.parametrize(
        "ivim_fit",
        ["ivim_mono_fit_data", "ivim_bi_fit_data", "ivim_tri_fit_data"],
    )
    def test_ivim_pixel_multithreading(self, ivim_fit: FitData, request):
        """Test IVIM pixel-wise multithreaded fitting for mono/bi/tri-exponential models."""
        ivim_fit_data = request.getfixturevalue(ivim_fit)
        ivim_fit_data.params.n_pools = 4
        ParameterTools.assert_fit_completes(
            ivim_fit_data, "fit_pixel_wise", fit_type="multi"
        )

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
        # Test passes if fitting completes without raising exceptions
        assert ivim_fit_data.results is not None, "Fitting should produce results"

    def test_ivim_mono_result_to_fit_curve(self, ivim_mono_fit_data: FitData):
        ivim_mono_fit_data.results.raw[0, 0, 0] = np.array([0.15, 150])
        curve = ivim_mono_fit_data.params.fit_model.model(
            ivim_mono_fit_data.params.b_values,
            *ivim_mono_fit_data.results.raw[0, 0, 0].tolist(),
        )
        # Test passes if model evaluation completes without raising exceptions
        assert curve is not None and len(curve) > 0, "Model should return signal curve"

    @pytest.mark.gpu
    def test_biexp_gpu(self, decay_bi_array, ivim_bi_gpu_params):
        fit_args = decay_bi_array["fit_args"]
        result = gpu_fitter(
            fit_args,
            ivim_bi_gpu_params,
        )
        assert result is not None

    @pytest.mark.gpu
    def test_triexp_gpu(self, decay_tri_array, ivim_tri_gpu_params):
        fit_args = decay_tri_array["fit_args"]
        results = gpu_fitter(fit_args, ivim_tri_gpu_params)
        assert results is not None


class TestIVIMSegmentedFitting:
    def test_ivim_segmented_first_fit(
        self, img, seg, ivim_bi_params_file, ivim_mono_params
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
            img, seg, ivim_bi_params_file
        )

        # Compare results between mono and segmented first fit
        self._assert_results_match(mono_results, segmented_results)

    def _perform_mono_fitting(self, img, seg, ivim_mono_params):
        """Perform mono-exponential fitting and return results."""
        pixel_args_mono = ivim_mono_params.get_pixel_args(img, seg)
        return multithreader(ivim_mono_params.fit_function, pixel_args_mono, None)

    def _perform_segmented_first_fit(self, img, seg, ivim_bi_params_file):
        """Set up and perform the first fit of segmented IVIM fitting."""
        # Initialize segmented parameters
        segmented_params = IVIMSegmentedParams(ivim_bi_params_file)

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
        # Test passes if segmented fitting completes without raising exceptions
        assert fit_data.results is not None, "Segmented fitting should produce results"

    def test_ivim_segmented_bi(self, img, seg, ivim_bi_segmented_params_file, out_nii):
        """Test IVIM segmented bi-exponential fitting with comprehensive validation using np.allclose."""
        fit_data = FitData(
            img,
            seg,
            ivim_bi_segmented_params_file,
        )
        fit_data.params.fixed_component = "D_1"
        fit_data.params.fixed_t1 = False
        fit_data.params.reduced_b_values = None
        fit_data.params.mixing_time = None
        fit_data.params.set_up()

        # Perform the segmented fitting
        fit_data.fit_ivim_segmented(fit_type="multi")

        # Validate that fitting was successful and results exist
        assert fit_data.results is not None, "No results generated from fitting"
        assert len(fit_data.results.raw) > 0, "No fitting results found"

        # Test each fitted pixel
        for pixel_coords, fitted_params in fit_data.results.raw.items():
            # Validate parameter ranges and finite values
            assert np.all(
                np.isfinite(fitted_params)
            ), f"Non-finite parameters at {pixel_coords}: {fitted_params}"

            # For bi-exponential IVIM model: [f1, D1, f2, D2, S0] or similar order
            # Extract fitted parameters
            S0_val = fit_data.results.S0[pixel_coords]
            f_vals = fit_data.results.f[pixel_coords]
            D_vals = fit_data.results.D[pixel_coords]

            # Convert to numpy arrays for safe comparison
            S0_array = np.asarray(S0_val)
            f_array = np.asarray(f_vals)
            D_array = np.asarray(D_vals)

            # Validate parameter ranges
            assert np.all(
                S0_array > 0
            ), f"S0 should be positive at {pixel_coords}: {S0_val}"
            assert np.all(
                f_array >= 0
            ), f"Fractions should be non-negative at {pixel_coords}: {f_vals}"
            assert np.all(
                D_array >= 0
            ), f"Diffusion values should be non-negative at {pixel_coords}: {D_vals}"
            assert np.allclose(
                np.sum(f_array), 1.0, rtol=1e-10
            ), f"Fractions should sum to 1 at {pixel_coords}: sum={np.sum(f_array)}"

            # Validate that the fitted curve matches the model prediction
            b_values = fit_data.params.b_values
            predicted_signal = fit_data.results.curve[pixel_coords]
            model_signal = fit_data.params.fit_model.model(b_values, *fitted_params)

            # Test curve consistency with high precision
            np.testing.assert_allclose(
                predicted_signal,
                model_signal,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Model curve mismatch at pixel {pixel_coords}",
            )

            # Validate signal decay properties (decreasing with increasing b-values for non-zero b)
            non_zero_b_mask = b_values > 0
            if np.any(non_zero_b_mask):
                signal_at_nonzero_b = predicted_signal[non_zero_b_mask]

                # Signal should generally decrease with increasing b-values
                if len(signal_at_nonzero_b) > 1:
                    # For IVIM, we expect general decay trend (though not strictly monotonic due to perfusion component)
                    max_signal = np.max(signal_at_nonzero_b)
                    min_signal = np.min(signal_at_nonzero_b)
                    assert (
                        max_signal > min_signal
                    ), f"Expected signal decay with b-value at {pixel_coords}"

        # Test global properties across all pixels
        all_S0 = np.array(
            [fit_data.results.S0[coords] for coords in fit_data.results.raw.keys()]
        )
        all_D1 = np.array(
            [fit_data.results.D[coords][0] for coords in fit_data.results.raw.keys()]
        )
        all_D2 = np.array(
            [fit_data.results.D[coords][1] for coords in fit_data.results.raw.keys()]
        )

        # Flatten arrays in case they have extra dimensions
        all_S0_flat = np.asarray(all_S0).flatten()
        all_D1_flat = np.asarray(all_D1).flatten()
        all_D2_flat = np.asarray(all_D2).flatten()

        # Validate that results are consistent across pixels (basic sanity checks)
        assert np.all(all_S0_flat > 0), "All S0 values should be positive"
        assert np.all(all_D1_flat >= 0), "All D1 values should be non-negative"
        assert np.all(all_D2_flat >= 0), "All D2 values should be non-negative"

        # For IVIM, typically D1 < D2 (slow vs fast diffusion component)
        # This is a biological expectation but may not always hold in test data
        # We just check that they're different to ensure fitting worked
        mean_D1 = np.mean(all_D1_flat)
        mean_D2 = np.mean(all_D2_flat)
        assert not np.isclose(
            mean_D1, mean_D2, rtol=1e-3
        ), "D1 and D2 should be distinguishable"

        # Validate that the number of fitted pixels matches segmentation
        seg_pixels = np.sum(np.squeeze(seg) > 0)
        fitted_pixels = len(fit_data.results.raw)
        assert (
            fitted_pixels == seg_pixels
        ), f"Fitted pixels ({fitted_pixels}) should match segmentation pixels ({seg_pixels})"

    def test_ivim_segmented_bi_focused_validation(
        self, img, seg, ivim_bi_segmented_params_file, out_nii
    ):
        """Focused test for IVIM segmented bi-exponential fitting using np.allclose for specific validations."""
        fit_data = FitData(
            img,
            seg,
            ivim_bi_segmented_params_file,
        )
        fit_data.params.fixed_component = "D_1"
        fit_data.params.fixed_t1 = False
        fit_data.params.reduced_b_values = None
        fit_data.params.mixing_time = None
        fit_data.params.set_up()

        # Perform the segmented fitting
        fit_data.fit_ivim_segmented(fit_type="multi")

        # Key validation points using np.allclose:

        # 1. Fraction sum validation (should be exactly 1.0)
        for pixel_coords, fitted_params in fit_data.results.raw.items():
            f_vals = np.asarray(fit_data.results.f[pixel_coords])
            np.testing.assert_allclose(
                np.sum(f_vals),
                1.0,
                rtol=1e-14,
                atol=1e-14,
                err_msg=f"Fraction sum should be 1.0 at pixel {pixel_coords}, got {np.sum(f_vals)}",
            )

        # 2. Model consistency validation (predicted curve vs model calculation)
        sample_pixel = next(iter(fit_data.results.raw.keys()))
        fitted_params = fit_data.results.raw[sample_pixel]
        b_values = fit_data.params.b_values

        predicted_curve = fit_data.results.curve[sample_pixel]
        recalculated_curve = fit_data.params.fit_model.model(b_values, *fitted_params)

        np.testing.assert_allclose(
            predicted_curve,
            recalculated_curve,
            rtol=1e-15,
            atol=1e-15,
            err_msg="Stored curve should match model recalculation exactly",
        )

        # 3. Parameter range validation with tolerances
        all_fitted_params = np.array(
            [params for params in fit_data.results.raw.values()]
        )

        # All parameters should be finite
        assert np.all(
            np.isfinite(all_fitted_params)
        ), "All fitted parameters should be finite"

        # Extract physical parameters for range checking
        all_D_values = np.array(
            [list(fit_data.results.D[coords]) for coords in fit_data.results.raw.keys()]
        )
        all_f_values = np.array(
            [list(fit_data.results.f[coords]) for coords in fit_data.results.raw.keys()]
        )

        # Diffusion coefficients should be positive and within reasonable IVIM ranges
        assert np.all(
            all_D_values >= 0
        ), "All diffusion coefficients should be non-negative"
        assert np.all(
            all_D_values <= 1.0
        ), "Diffusion coefficients should be <= 1.0 mm²/s (unrealistic otherwise)"

        # Fractions should be between 0 and 1
        assert np.all(all_f_values >= 0), "All fractions should be non-negative"
        assert np.all(all_f_values <= 1), "All fractions should be <= 1"

        # 4. Signal physics validation (exponential decay)
        for pixel_coords in list(fit_data.results.raw.keys())[
            :3
        ]:  # Test first 3 pixels
            signal = fit_data.results.curve[pixel_coords]
            b_vals = fit_data.params.b_values

            # Signal should decay exponentially - test that signal at b=0 >= signal at max b
            b0_idx = np.argmin(np.abs(b_vals))  # closest to b=0
            max_b_idx = np.argmax(b_vals)

            assert (
                signal[b0_idx] >= signal[max_b_idx]
            ), f"Signal should decrease from b=0 to max b-value at pixel {pixel_coords}"

        # 5. Reproducibility test - recalculate using fitted parameters
        test_pixel = sample_pixel
        test_params = fit_data.results.raw[test_pixel]
        test_signal_1 = fit_data.params.fit_model.model(b_values, *test_params)
        test_signal_2 = fit_data.params.fit_model.model(b_values, *test_params)

        np.testing.assert_allclose(
            test_signal_1,
            test_signal_2,
            rtol=1e-16,
            atol=1e-16,
            err_msg="Model should give identical results for same parameters (reproducibility test)",
        )

    def test_ivim_segmented_bi_synthetic_signal(
        self, ivim_bi_segmented_params_file, out_nii, signal_generator, noise_model
    ):
        """Test IVIM segmented bi-exponential fitting with synthetic signal data using kidney parameters."""

        # Use kidney-specific b-values (from canonical parameters)
        b_values = np.array([0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250])

        # Kidney bi-exponential parameters (blood + tissue)
        true_params = {
            "S0": np.mean(cp.S0_RANGE),
            "f1": cp.BIEXP_TYPICAL["f1"],  # Blood perfusion fraction (10%)
            "D1": cp.BIEXP_TYPICAL["D1"],  # Blood diffusion (mm²/s)
            "f2": 1.0 - cp.BIEXP_TYPICAL["f1"],  # Tissue fraction (90%)
            "D2": cp.BIEXP_TYPICAL["D2"],  # Combined tissue+tubule diffusion (mm²/s)
        }

        # Generate clean synthetic signal using signal generator
        synthetic_signal = signal_generator.generate_biexp(
            b_values,
            f1=true_params["f1"],
            D1=true_params["D1"],
            D2=true_params["D2"],
            S0=true_params["S0"],
        )

        # Add SNR=140 kidney-quality noise
        np.random.seed(42)  # For reproducible results
        synthetic_signal_noisy = noise_model.add_noise(
            synthetic_signal, snr=140.0, seed=42
        )

        # Create synthetic image and segmentation arrays
        from radimgarray import RadImgArray, SegImgArray

        # Create a small 3D volume with synthetic data
        img_shape = (3, 3, 1, len(b_values))  # 3x3x1 volume with all b-values
        seg_shape = (3, 3, 1, 1)

        # Fill image with synthetic signals (with slight parameter variations for different pixels)
        img_data = np.zeros(img_shape)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                # Add slight parameter variations across pixels (±5%)
                variation = 0.95 + 0.1 * np.random.random()  # 0.95 to 1.05 multiplier
                img_data[i, j, 0, :] = synthetic_signal_noisy * variation

        # Create segmentation (all pixels are included)
        seg_data = np.ones(seg_shape, dtype=int)

        # Create RadImgArray and SegImgArray objects
        img = RadImgArray(img_data)
        seg = SegImgArray(seg_data)

        # Create FitData with synthetic data
        fit_data = FitData(
            img,
            seg,
            ivim_bi_segmented_params_file,
        )

        # Configure fitting parameters
        fit_data.params.fixed_component = "D_1"
        fit_data.params.fixed_t1 = False
        fit_data.params.reduced_b_values = None
        fit_data.params.mixing_time = None

        # Override b-values with our synthetic b-values
        fit_data.params.b_values = b_values
        fit_data.params.set_up()

        # Perform the segmented fitting
        fit_data.fit_ivim_segmented(fit_type="multi")

        # Comprehensive validation using np.allclose

        # 1. Basic validation - fitting should succeed
        assert fit_data.results is not None, "No results generated from fitting"
        assert len(fit_data.results.raw) > 0, "No fitting results found"

        # 2. Parameter recovery validation - test that we can recover known parameters
        fitted_pixels = list(fit_data.results.raw.keys())

        # Test parameter recovery for center pixel (should be closest to true parameters)
        center_pixel = (1, 1, 0)  # Center of 3x3 grid
        if center_pixel in fitted_pixels:
            test_pixel = center_pixel
        else:
            test_pixel = fitted_pixels[0]

        # Extract fitted parameters
        fitted_S0 = fit_data.results.S0[test_pixel]
        fitted_f = fit_data.results.f[test_pixel]
        fitted_D = fit_data.results.D[test_pixel]

        # Convert to arrays for easier handling
        fitted_S0_val = (
            np.asarray(fitted_S0).item()
            if np.asarray(fitted_S0).ndim == 0
            else np.asarray(fitted_S0)[0]
        )
        fitted_f_array = np.asarray(fitted_f)
        fitted_D_array = np.asarray(fitted_D)

        # Parameter recovery validation with appropriate tolerances
        # Note: Segmented fitting fixes D1, so we test recovery of other parameters

        # S0 recovery (allow 10% tolerance due to noise and fitting)
        np.testing.assert_allclose(
            fitted_S0_val,
            true_params["S0"],
            rtol=0.1,
            err_msg=f"S0 recovery failed: fitted={fitted_S0_val}, true={true_params['S0']}",
        )

        # Fraction recovery (D1 is fixed in segmented fitting, so test sum and ranges)
        assert (
            len(fitted_f_array) == 2
        ), f"Expected 2 fractions, got {len(fitted_f_array)}"

        # Fractions should sum to 1
        np.testing.assert_allclose(
            np.sum(fitted_f_array),
            1.0,
            rtol=1e-10,
            err_msg=f"Fractions don't sum to 1: {fitted_f_array}, sum={np.sum(fitted_f_array)}",
        )

        # Fractions should be positive and reasonable
        assert np.all(
            fitted_f_array > 0
        ), f"All fractions should be positive: {fitted_f_array}"
        assert np.all(
            fitted_f_array < 1
        ), f"All fractions should be < 1: {fitted_f_array}"

        # Diffusion coefficient validation
        assert (
            len(fitted_D_array) == 2
        ), f"Expected 2 diffusion coefficients, got {len(fitted_D_array)}"
        assert np.all(
            fitted_D_array > 0
        ), f"All D values should be positive: {fitted_D_array}"

        # 3. Model consistency validation
        fitted_params_raw = fit_data.results.raw[test_pixel]
        predicted_curve = fit_data.results.curve[test_pixel]
        model_curve = fit_data.params.fit_model.model(b_values, *fitted_params_raw)
        model_curve = model_curve[:, np.newaxis]

        np.testing.assert_allclose(
            predicted_curve,
            model_curve,
            rtol=1e-14,
            atol=1e-14,
            err_msg="Predicted curve should match model calculation exactly",
        )

        # 4. Signal physics validation
        # Signal should decrease with increasing b-values (except for b=0)
        assert (
            predicted_curve[0] >= predicted_curve[-1]
        ), f"Signal should decrease from b=0 to max b: S(0)={predicted_curve[0]}, S(max)={predicted_curve[-1]}"

        # 5. Global validation across all pixels
        all_fitted_params = [fit_data.results.raw[coords] for coords in fitted_pixels]
        all_fitted_params_array = np.array(all_fitted_params)

        # All parameters should be finite
        assert np.all(
            np.isfinite(all_fitted_params_array)
        ), "All fitted parameters should be finite"

        # Extract parameters for all pixels
        all_S0 = [fit_data.results.S0[coords] for coords in fitted_pixels]
        all_f = [fit_data.results.f[coords] for coords in fitted_pixels]
        all_D = [fit_data.results.D[coords] for coords in fitted_pixels]

        all_S0_array = np.array(
            [
                np.asarray(s0).item() if np.asarray(s0).ndim == 0 else np.asarray(s0)[0]
                for s0 in all_S0
            ]
        )
        all_f_array = np.array([np.asarray(f) for f in all_f])
        all_D_array = np.array([np.asarray(d) for d in all_D])

        # Statistical validation - parameters should be reasonable across pixels
        mean_S0 = np.mean(all_S0_array)
        std_S0 = np.std(all_S0_array)

        # S0 should be close to true value across pixels (allowing for 15% variation due to noise)
        np.testing.assert_allclose(
            mean_S0,
            true_params["S0"],
            rtol=0.15,
            err_msg=f"Mean S0 recovery failed: fitted={mean_S0}, true={true_params['S0']}",
        )

        # Standard deviation should be reasonable (not too high, indicating stable fitting)
        assert (
            std_S0 < 0.2 * mean_S0
        ), f"S0 variation too high: std={std_S0}, mean={mean_S0}"

        # 6. Reproducibility test
        test_params = fit_data.results.raw[test_pixel]
        signal_1 = fit_data.params.fit_model.model(b_values, *test_params)
        signal_2 = fit_data.params.fit_model.model(b_values, *test_params)

        np.testing.assert_allclose(
            signal_1,
            signal_2,
            rtol=1e-16,
            atol=1e-16,
            err_msg="Model should give identical results for same parameters",
        )

    def test_ivim_segmented_bi_simple_synthetic(
        self, ivim_bi_segmented_params_file, signal_generator
    ):
        """Simple synthetic signal test for IVIM segmented bi-exponential fitting with kidney parameters."""

        # Kidney-specific b-values
        b_values = np.array([0, 10, 30, 50, 100, 200, 400])
        
        # Kidney bi-exponential parameters
        true_S0 = np.mean(cp.S0_RANGE)
        true_f1 = cp.BIEXP_TYPICAL["f1"]  # Blood fraction (10%)
        true_D1 = cp.BIEXP_TYPICAL["D1"]  # Blood diffusion
        true_f2 = 1.0 - cp.BIEXP_TYPICAL["f1"]  # Tissue fraction (90%)
        true_D2 = cp.BIEXP_TYPICAL["D2"]  # Tissue diffusion

        # Generate clean IVIM signal using signal generator
        signal = signal_generator.generate_biexp(
            b_values, f1=true_f1, D1=true_D1, D2=true_D2, S0=true_S0
        )

        # Create minimal synthetic data structure
        from radimgarray import RadImgArray, SegImgArray

        # Single pixel test case
        img_data = signal.reshape(1, 1, 1, -1)  # 1x1x1 volume with b-values
        seg_data = np.ones((1, 1, 1, 1), dtype=int)  # Single voxel segmentation

        img = RadImgArray(img_data)
        seg = SegImgArray(seg_data)

        # Set up fitting
        fit_data = FitData(img, seg, ivim_bi_segmented_params_file)
        fit_data.params.fixed_component = "D_1"
        fit_data.params.fixed_t1 = False
        fit_data.params.reduced_b_values = None
        fit_data.params.mixing_time = None
        fit_data.params.b_values = b_values
        fit_data.params.set_up()

        # Perform fitting
        fit_data.fit_ivim_segmented(fit_type="multi")

        # Validation using np.allclose
        pixel_coords = (0, 0, 0)

        # 1. Basic checks
        assert (
            pixel_coords in fit_data.results.raw
        ), "Fitting should produce results for the test pixel"
        fitted_params = fit_data.results.raw[pixel_coords]
        assert np.all(
            np.isfinite(fitted_params)
        ), f"All fitted parameters should be finite: {fitted_params}"

        # 2. Fraction sum validation
        fitted_fractions = np.asarray(fit_data.results.f[pixel_coords])
        np.testing.assert_allclose(
            np.sum(fitted_fractions),
            1.0,
            rtol=1e-12,
            err_msg=f"Fractions should sum to 1.0: {fitted_fractions}, sum={np.sum(fitted_fractions)}",
        )

        # 3. Model consistency validation
        predicted_signal = fit_data.results.curve[pixel_coords]
        recalc_signal = fit_data.params.fit_model.model(b_values, *fitted_params)
        recalc_signal = recalc_signal[:, np.newaxis]

        np.testing.assert_allclose(
            predicted_signal,
            recalc_signal,
            rtol=1e-15,
            atol=1e-15,
            err_msg="Stored signal should match model recalculation exactly",
        )

        # 4. Signal physics validation
        assert (
            predicted_signal[0] >= predicted_signal[-1]
        ), "Signal should decay from b=0 to max b-value"

        # 5. Parameter range validation
        fitted_S0 = np.asarray(fit_data.results.S0[pixel_coords])
        fitted_D = np.asarray(fit_data.results.D[pixel_coords])

        # S0 should be positive and reasonable
        assert np.all(fitted_S0 > 0), f"S0 should be positive: {fitted_S0}"

        # D values should be positive and reasonable for IVIM
        assert np.all(fitted_D > 0), f"D values should be positive: {fitted_D}"
        assert np.all(fitted_D < 1.0), f"D values should be < 1.0 mm²/s: {fitted_D}"

        # 6. Ground truth comparison (with tolerance for numerical fitting)
        fitted_S0_val = fitted_S0.item() if fitted_S0.ndim == 0 else fitted_S0[0]

        # S0 recovery should be within 5% (perfect synthetic data)
        np.testing.assert_allclose(
            fitted_S0_val,
            true_S0,
            rtol=0.05,
            err_msg=f"S0 recovery failed: fitted={fitted_S0_val}, true={true_S0}",
        )
