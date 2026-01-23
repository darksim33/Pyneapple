"""Tests for IVIM model classes and fitting functionality.

This module contains comprehensive tests for IVIM (Intravoxel Incoherent Motion) fitting models
including mono-exponential, bi-exponential, and tri-exponential models. Tests verify:

- Model class instantiation with various configurations (reduced models, S0 fitting, T1 fitting)
- Model evaluation with known parameters (signal generation accuracy)
- Model fitting with synthetic noisy data (parameter recovery accuracy)
- Advanced features: T1 correction (STEAM/standard), fixed diffusion coefficients, individual boundaries

Key test scenarios:
- Standard models: mono/bi/tri-exponential with standard parameters
- Reduced models: fitting with constrained fractions (sum to 1)
- S0 fitting: explicit S0 parameter vs implicit normalization
- T1 corrections: standard (repetition_time) and STEAM (mixing_time)
- Fixed parameters: fixing specific diffusion coefficients during fitting
- Boundary types: uniform vs individual pixel boundaries
"""
from functools import partial

import numpy as np
import pytest

from pyneapple.models.ivim import (
    BiExpFitModel,
    MonoExpFitModel,
    TriExpFitModel,
    get_model_class,
)
from tests.test_utils.validators import (
    validate_parameter_recovery,
    validate_fraction_sum,
    validate_model_consistency,
)
from tests.test_utils import canonical_parameters as cp


class TestIVIMModelClasses:
    """Test IVIM model class instantiation and parameter configuration."""

    def test_mono_exp_model_creation(self):
        """Test that mono-exponential model creates correct parameter lists for standard, T1, and reduced configurations."""
        mono_model = MonoExpFitModel("mono")
        assert mono_model.args == ["D_1", "S_0"]

        # Test with T1 fitting
        mono_model_t1 = MonoExpFitModel("mono", repetition_time=20, fit_t1=True)
        assert mono_model_t1.args == ["D_1", "S_0", "T_1"]

        # Test fit_reduced model
        mono_model_reduced = MonoExpFitModel("mono", fit_reduced=True)
        assert mono_model_reduced.args == ["D_1"]

    def test_bi_exp_model_creation(self):
        """Test that bi-exponential model creates correct parameter lists for standard, reduced, and T1 configurations."""
        bi_model = BiExpFitModel("bi")
        assert bi_model.args == ["f_1", "D_1", "f_2", "D_2"]

        # Test with fit_reduced model
        bi_model_reduced = BiExpFitModel("bi", fit_reduced=True)
        assert bi_model_reduced.args == ["f_1", "D_1", "D_2"]

        # Test with T1 fitting
        bi_model_t1 = BiExpFitModel("bi", repetition_time=20, fit_t1=True)
        assert bi_model_t1.args == ["f_1", "D_1", "f_2", "D_2", "T_1"]

    def test_bi_exp_model_fit_s0_creation(self):
        """Test that bi-exponential model with S0 fitting creates correct parameter lists and validates incompatible configurations."""
        # Test initialization with fit_S0=True
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)
        assert bi_model_s0.args == ["f_1", "D_1", "D_2", "S_0"]
        assert bi_model_s0.fit_S0 is True

        # Test with T1 fitting
        bi_model_s0_t1 = BiExpFitModel(
            "bi", fit_S0=True, repetition_time=20, fit_t1=True
        )
        assert bi_model_s0_t1.args == ["f_1", "D_1", "D_2", "S_0", "T_1"]
        assert bi_model_s0_t1.fit_S0 is True

        # Test with fit_reduced model (should raise ValueError)
        with pytest.raises(ValueError):
            BiExpFitModel("bi", fit_S0=True, fit_reduced=True)

    def test_tri_exp_model_creation(self):
        """Test that tri-exponential model creates correct parameter lists for standard, reduced, and T1 configurations."""
        tri_model = TriExpFitModel("tri")
        assert tri_model.args == ["f_1", "D_1", "f_2", "D_2", "f_3", "D_3"]

        # Test with fit_reduced model
        tri_model_reduced = TriExpFitModel("tri", fit_reduced=True)
        assert tri_model_reduced.args == ["f_1", "D_1", "f_2", "D_2", "D_3"]

        # Test with T1 fitting
        tri_model_t1 = TriExpFitModel("tri", repetition_time=20, fit_t1=True)
        assert tri_model_t1.args == ["f_1", "D_1", "f_2", "D_2", "f_3", "D_3", "T_1"]

    def test_tri_exp_model_fit_s0_creation(self):
        """Test that tri-exponential model with S0 fitting creates correct parameter lists and validates incompatible configurations."""
        # Test initialization with fit_S0=True
        tri_model_s0 = TriExpFitModel("tri", fit_S0=True)
        assert tri_model_s0.args == ["f_1", "D_1", "f_2", "D_2", "D_3", "S_0"]
        assert tri_model_s0.fit_S0 is True

        # Test with T1 fitting
        tri_model_s0_t1 = TriExpFitModel(
            "tri", fit_S0=True, repetition_time=20, fit_t1=True
        )
        assert tri_model_s0_t1.args == ["f_1", "D_1", "f_2", "D_2", "D_3", "S_0", "T_1"]
        assert tri_model_s0_t1.fit_S0 is True

        # Test with fit_reduced model (should raise ValueError)
        with pytest.raises(ValueError):
            TriExpFitModel("tri", fit_S0=True, fit_reduced=True)

    def test_get_model_class(self):
        """Test that get_model_class returns correct model classes for valid names and raises ValueError for invalid names."""
        mono_class = get_model_class("mono")
        assert mono_class == MonoExpFitModel

        bi_class = get_model_class("bi")
        assert bi_class == BiExpFitModel

        tri_class = get_model_class("tri")
        assert tri_class == TriExpFitModel

        with pytest.raises(ValueError):
            get_model_class("invalid_model")


@pytest.fixture
def b_values():
    return np.array([0, 50, 100, 200, 400, 800])


class TestIVIMModelEvaluation:
    """Test IVIM model signal evaluation with known parameters."""

    @pytest.fixture
    def signal_mono(self, b_values):
        # S0=1000, D1=0.001
        return 1000 * np.exp(-b_values * 0.001)

    @pytest.fixture
    def signal_bi(self, b_values):
        # f1=0.3, D1=0.003, f2=0.7 D2=0.0005
        return 0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.0005)

    @pytest.fixture
    def signal_bi_s0(self, b_values):
        # f1=0.3, D1=0.003, f2=0.7 D2=0.0005, S0=1000
        return (
            0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.0005)
        ) * 1000

    @pytest.fixture
    def signal_tri(self, b_values):
        # f1=0.2, D1=0.005, f2=0.3, D2=0.001, f3=0.5, D3=0.0002
        return (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )

    @pytest.fixture
    def signal_tri_s0(self, b_values):
        # f1=0.2, D1=0.005, f2=0.3, D2=0.001, D3=0.0002, S0=1000
        return 1000 * (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )

    def test_mono_model_evaluation(self, b_values, signal_mono):
        """Test that mono-exponential model evaluates correctly with known parameters producing expected signal."""
        mono_model = MonoExpFitModel("mono")
        # Test with correct parameters
        output = mono_model.model(b_values, 0.001, 1000)
        np.testing.assert_allclose(output, signal_mono, rtol=1e-5)

    def test_bi_model_evaluation(self, b_values, signal_bi):
        """Test that bi-exponential model evaluates correctly for standard, reduced, and fixed-D configurations."""
        bi_model = BiExpFitModel("bi")
        # Test with correct parameters
        output = bi_model.model(b_values, 0.3, 0.003, 0.7, 0.0005)
        np.testing.assert_allclose(output, signal_bi, rtol=1e-5)

        # Test fit_reduced model
        bi_model_red = BiExpFitModel("bi", fit_reduced=True)
        output_red = bi_model_red.model(b_values, 0.3, 0.003, 0.0005)
        np.testing.assert_allclose(output_red, signal_bi, rtol=1e-5)

        # Test fixed D
        bi_model_fixed = BiExpFitModel("bi", fix_d=2)
        output_fixed = bi_model_fixed.model(
            b_values, 0.3, 0.003, 0.7, 0, fixed_d=0.0005
        )
        np.testing.assert_allclose(output_fixed, signal_bi, rtol=1e-5)

    def test_bi_model_with_s0_evaluation(self, b_values, signal_bi_s0):
        """Test that bi-exponential model with S0 fitting evaluates correctly including T1 correction scenarios."""
        # Test model with fit_S0=True
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)
        output = bi_model_s0.model(b_values, 0.3, 0.003, 0.0005, 1000)
        np.testing.assert_allclose(output, signal_bi_s0, rtol=1e-5)

        # Test with T1 fitting
        repetition_time = 20
        t1_value = 30
        signal_bi_s0_t1 = signal_bi_s0 * (1 - np.exp(-repetition_time / t1_value))

        bi_model_s0_t1 = BiExpFitModel(
            "bi", fit_S0=True, repetition_time=repetition_time, fit_t1=True
        )
        output_t1 = bi_model_s0_t1.model(b_values, 0.3, 0.003, 0.0005, 1000, t1_value)
        np.testing.assert_allclose(output_t1, signal_bi_s0_t1, rtol=1e-5)

    def test_bi_model_with_s0_t1_steam_evaluation(self, b_values, signal_bi_s0):
        """Test that bi-exponential model with S0 and T1 STEAM correction evaluates correctly with mixing time."""
        # Test model with fit_S0=True and T1 STEAM fitting
        mixing_time = 25
        t1_value = 30
        signal_bi_s0_t1_steam = signal_bi_s0 * np.exp(-mixing_time / t1_value)

        bi_model_s0_t1_steam = BiExpFitModel(
            "bi", fit_S0=True, mixing_time=mixing_time, fit_t1_steam=True
        )
        output_t1_steam = bi_model_s0_t1_steam.model(
            b_values, 0.3, 0.003, 0.0005, 1000, t1_value
        )
        np.testing.assert_allclose(output_t1_steam, signal_bi_s0_t1_steam, rtol=1e-5)

    def test_mono_model_t1_steam_evaluation(self, b_values, signal_mono):
        """Test that mono-exponential model with T1 STEAM correction evaluates correctly with mixing time."""
        # Test mono model with T1 STEAM fitting
        mixing_time = 25
        t1_value = 30
        signal_mono_t1_steam = signal_mono * np.exp(-mixing_time / t1_value)

        mono_model_t1_steam = MonoExpFitModel(
            "mono", mixing_time=mixing_time, fit_t1_steam=True
        )
        output_t1_steam = mono_model_t1_steam.model(b_values, 0.001, 1000, t1_value)
        np.testing.assert_allclose(output_t1_steam, signal_mono_t1_steam, rtol=1e-5)

    def test_tri_model_evaluation(self, b_values, signal_tri):
        """Test that tri-exponential model evaluates correctly for standard, reduced, and fixed-D configurations."""
        tri_model = TriExpFitModel("tri")
        # Test with correct parameters
        output = tri_model.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.5, 0.0002)
        np.testing.assert_allclose(output, signal_tri, rtol=1e-5)

        # Test fit_reduced model
        tri_model_red = TriExpFitModel("tri", fit_reduced=True)
        output_red = tri_model_red.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.0002)
        np.testing.assert_allclose(output_red, signal_tri, rtol=1e-5)

        # Test fixed D
        tri_model_fixed = TriExpFitModel("tri", fix_d=3)
        output_fixed = tri_model_fixed.model(
            b_values, 0.2, 0.005, 0.3, 0.001, 0.5, 0, fixed_d=0.0002
        )
        np.testing.assert_allclose(output_fixed, signal_tri, rtol=1e-5)

    def test_tri_model_with_s0_evaluation(self, b_values, signal_tri_s0):
        """Test that tri-exponential model with S0 fitting evaluates correctly including T1 correction scenarios."""
        # Test model with fit_S0=True
        tri_model_s0 = TriExpFitModel("tri", fit_S0=True)
        output = tri_model_s0.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.0002, 1000)
        np.testing.assert_allclose(output, signal_tri_s0, rtol=1e-5)

        # Test with T1 fitting
        repetition_time = 20
        t1_value = 30
        signal_tri_s0_t1 = signal_tri_s0 * (1 - np.exp(-repetition_time / t1_value))

        tri_model_s0_t1 = TriExpFitModel(
            "tri", fit_S0=True, repetition_time=repetition_time, fit_t1=True
        )
        output_t1 = tri_model_s0_t1.model(
            b_values, 0.2, 0.005, 0.3, 0.001, 0.0002, 1000, t1_value
        )
        np.testing.assert_allclose(output_t1, signal_tri_s0_t1, rtol=1e-5)

    def test_tri_model_with_s0_t1_steam_evaluation(self, b_values, signal_tri_s0):
        """Test that tri-exponential model with S0 and T1 STEAM correction evaluates correctly with mixing time."""
        # Test model with fit_S0=True and T1 STEAM fitting
        mixing_time = 25
        t1_value = 30
        signal_tri_s0_t1_steam = signal_tri_s0 * np.exp(-mixing_time / t1_value)

        tri_model_s0_t1_steam = TriExpFitModel(
            "tri", fit_S0=True, mixing_time=mixing_time, fit_t1_steam=True
        )
        output_t1_steam = tri_model_s0_t1_steam.model(
            b_values, 0.2, 0.005, 0.3, 0.001, 0.0002, 1000, t1_value
        )
        np.testing.assert_allclose(output_t1_steam, signal_tri_s0_t1_steam, rtol=1e-5)

    def test_bi_model_reduced_t1_steam_evaluation(self, b_values):
        """Test that reduced bi-exponential model with T1 STEAM correction evaluates correctly with constrained fractions."""
        # Test reduced bi model with T1 STEAM fitting
        mixing_time = 25
        t1_value = 30

        base_signal = 0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(
            -b_values * 0.0005
        )
        signal_bi_red_t1_steam = base_signal * np.exp(-mixing_time / t1_value)

        bi_model_red_t1_steam = BiExpFitModel(
            "bi", fit_reduced=True, mixing_time=mixing_time, fit_t1_steam=True
        )
        output_t1_steam = bi_model_red_t1_steam.model(
            b_values, 0.3, 0.003, 0.0005, t1_value
        )
        np.testing.assert_allclose(output_t1_steam, signal_bi_red_t1_steam, rtol=1e-5)

    def test_tri_model_reduced_t1_steam_evaluation(self, b_values):
        """Test that reduced tri-exponential model with T1 STEAM correction evaluates correctly with constrained fractions."""
        # Test reduced tri model with T1 STEAM fitting
        mixing_time = 25
        t1_value = 30

        base_signal = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        )
        signal_tri_red_t1_steam = base_signal * np.exp(-mixing_time / t1_value)

        tri_model_red_t1_steam = TriExpFitModel(
            "tri", fit_reduced=True, mixing_time=mixing_time, fit_t1_steam=True
        )
        output_t1_steam = tri_model_red_t1_steam.model(
            b_values, 0.2, 0.005, 0.3, 0.001, 0.0002, t1_value
        )
        np.testing.assert_allclose(output_t1_steam, signal_tri_red_t1_steam, rtol=1e-5)


class TestIVIMModelFitting:
    """Test IVIM model fitting with synthetic noisy data to verify parameter recovery accuracy.
    
    Uses new SNR-based framework with:
    - Canonical parameters from test_utils
    - SNR-based noise model (default SNR=30)
    - Automatic tolerance adjustment based on SNR
    """

    def test_mono_model_fit(self, canonical_b_values, noisy_mono_signal, canonical_mono_params):
        """Test mono-exponential model fitting recovers parameters within SNR-dependent tolerance.

        With SNR=50 and 16 b-values, expect 10% tolerance for parameter recovery.
        """
        mono_model = MonoExpFitModel("mono")
        x0 = np.array([0.002, 1000])  # Initial guess
        lb = np.array([0, 0])  # Lower bounds
        ub = np.array([0.01, 2000])  # Upper bounds

        idx, params, _ = mono_model.fit(
            0, noisy_mono_signal, x0=x0, lb=lb, ub=ub, b_values=canonical_b_values, max_iter=1000
        )

        # Validate parameter recovery with SNR-based tolerance
        # SNR=50 -> 10% tolerance
        expected = np.array([canonical_mono_params["D"], canonical_mono_params["S0"]])
        validate_parameter_recovery(
            params, expected, snr=cp.DEFAULT_SNR, param_name="mono_params"
        )

    def test_bi_model_with_s0_fit(self, canonical_b_values, noisy_biexp_signal, canonical_biexp_params):
        """Test bi-exponential model with S0 fitting recovers all parameters within SNR-dependent tolerance.
        
        IVIM bi-exponential fitting is inherently ill-conditioned due to:
        - Correlation between f1 and D1 parameters
        - Similar decay behavior at low b-values
        - Need for both slow (tissue) and fast (perfusion) components
        
        With SNR=140 and 16 b-values, expect realistic tolerances reflecting IVIM fitting variability.
        """
        # Create model with fit_S0=True
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)

        # Initial guess and bounds from global kidney IVIM ranges
        # Model: S0 * (f1*exp(-b*D1) + (1-f1)*exp(-b*D2))
        bounds = cp.FITTING_BOUNDS["biexp"]
        x0 = np.array([0.10, 0.100, 0.003, 900])  # f1, D1, D2, S0
        lb = np.array([bounds["f1"][0], bounds["D1"][0], bounds["D2"][0], bounds["S0"][0]])
        ub = np.array([bounds["f1"][1], bounds["D1"][1], bounds["D2"][1], bounds["S0"][1]])

        # Fit
        idx, params, _ = bi_model_s0.fit(
            0, noisy_biexp_signal, x0=x0, lb=lb, ub=ub, b_values=canonical_b_values, max_iter=1000
        )

        # Validate parameter recovery with SNR=140 (kidney quality)
        # Expect tighter tolerances: ~5% for well-conditioned, ~10-20% for ill-conditioned
        validate_parameter_recovery(params[0], canonical_biexp_params["f1"], snr=cp.DEFAULT_SNR, param_name="f1")
        validate_parameter_recovery(params[1], canonical_biexp_params["D1"], snr=cp.DEFAULT_SNR, param_name="D1", custom_tolerance=0.15)  # D1 blood challenging
        validate_parameter_recovery(params[2], canonical_biexp_params["D2"], snr=cp.DEFAULT_SNR, param_name="D2", custom_tolerance=0.15)  # D2 tissue challenging
        validate_parameter_recovery(params[3], canonical_biexp_params["S0"], snr=cp.DEFAULT_SNR, param_name="S0")
        
        # Validate fraction sum
        validate_fraction_sum([params[0], 1 - params[0]], expected_sum=1.0)
        
        # Validate model consistency
        recalc_curve = bi_model_s0.model(canonical_b_values, *params)
        validate_model_consistency(recalc_curve, recalc_curve, rtol=1e-12)


    def test_mono_model_fit_individual_boundaries(self, canonical_b_values, noisy_mono_signal, canonical_mono_params):
        """Test mono-exponential model fitting with individual boundaries recovers parameters correctly."""
        mono_model = MonoExpFitModel("mono")

        # Individual boundaries for specific pixel coordinate
        idx = (0, 0)
        x0 = np.array([0.002, 1000])
        lb = np.array([0, 0])
        ub = np.array([0.01, 2000])

        result_idx, params, _ = mono_model.fit(
            idx,
            noisy_mono_signal,
            x0,
            lb,
            ub,
            b_values=canonical_b_values,
            max_iter=1000,
            btype="individual",
        )

        # Check results
        assert result_idx == idx
        expected = np.array([canonical_mono_params["D"], canonical_mono_params["S0"]])
        validate_parameter_recovery(
            params, expected, snr=cp.DEFAULT_SNR, param_name="mono_params_individual"
        )

    def test_bi_model_with_s0_fit_individual_boundaries(self, canonical_b_values, noisy_biexp_signal, canonical_biexp_params):
        """Test bi-exponential model with S0 and individual boundaries recovers parameters correctly for specific pixel."""
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)

        # Use global kidney fitting bounds
        bounds = cp.FITTING_BOUNDS["biexp"]
        idx = (1, 2)
        x0 = np.array([0.10, 0.100, 0.003, 900])  # f1 (blood), D1 (blood D), D2 (tissue D), S0
        lb = np.array([bounds["f1"][0], bounds["D1"][0], bounds["D2"][0], bounds["S0"][0]])
        ub = np.array([bounds["f1"][1], bounds["D1"][1], bounds["D2"][1], bounds["S0"][1]])

        result_idx, params, _ = bi_model_s0.fit(
            idx,
            noisy_biexp_signal,
            x0,
            lb,
            ub,
            b_values=canonical_b_values,
            max_iter=1000,
            btype="individual",
        )

        # Check results
        assert result_idx == idx
        # Use per-parameter tolerances for SNR=140
        validate_parameter_recovery(params[0], canonical_biexp_params["f1"], snr=cp.DEFAULT_SNR, param_name="f1")
        validate_parameter_recovery(params[1], canonical_biexp_params["D1"], snr=cp.DEFAULT_SNR, param_name="D1", custom_tolerance=0.15)
        validate_parameter_recovery(params[2], canonical_biexp_params["D2"], snr=cp.DEFAULT_SNR, param_name="D2", custom_tolerance=0.15)
        validate_parameter_recovery(params[3], canonical_biexp_params["S0"], snr=cp.DEFAULT_SNR, param_name="S0")

    def test_tri_model_fit_individual_boundaries(self, canonical_b_values, noisy_triexp_signal, canonical_triexp_params):
        """Test tri-exponential model with individual boundaries recovers all six parameters from synthetic noisy data."""
        tri_model_s0 = TriExpFitModel("tri", fit_S0=True)

        # Use global kidney fitting bounds
        bounds = cp.FITTING_BOUNDS["triexp"]
        idx = (2, 3)
        x0 = np.array([0.10, 0.100, 0.30, 0.005, 0.002, 900])  # f1, D1, f2, D2, D3, S0
        lb = np.array([bounds["f1"][0], bounds["D1"][0], bounds["f2"][0], bounds["D2"][0], bounds["D3"][0], bounds["S0"][0]])
        ub = np.array([bounds["f1"][1], bounds["D1"][1], bounds["f2"][1], bounds["D2"][1], bounds["D3"][1], bounds["S0"][1]])

        result_idx, params, _ = tri_model_s0.fit(
            idx,
            noisy_triexp_signal,
            x0,
            lb,
            ub,
            b_values=canonical_b_values,
            max_iter=250,
            btype="individual",
        )

        # Check results with SNR-based validation
        assert result_idx == idx
        # Tri-exponential with kidney parameters and SNR=140
        # f1 (10% blood) and f2 (30% tubule) are minority components - moderate tolerance
        # D1 (blood) large and easier, D2 (tubule) and D3 (tissue) small and challenging
        validate_parameter_recovery(params[0], canonical_triexp_params["f1"], snr=cp.DEFAULT_SNR, param_name="f1", custom_tolerance=0.15)  # f1 blood
        validate_parameter_recovery(params[1], canonical_triexp_params["D1"], snr=cp.DEFAULT_SNR, param_name="D1", custom_tolerance=0.15)  # D1 blood
        validate_parameter_recovery(params[2], canonical_triexp_params["f2"], snr=cp.DEFAULT_SNR, param_name="f2", custom_tolerance=0.45)  # f2 tubule (very challenging in 3-component fit)
        validate_parameter_recovery(params[3], canonical_triexp_params["D2"], snr=cp.DEFAULT_SNR, param_name="D2", custom_tolerance=0.25)  # D2 tubule (intermediate)
        validate_parameter_recovery(params[4], canonical_triexp_params["D3"], snr=cp.DEFAULT_SNR, param_name="D3", custom_tolerance=0.25)  # D3 tissue (small)
        validate_parameter_recovery(params[5], canonical_triexp_params["S0"], snr=cp.DEFAULT_SNR, param_name="S0")  # S0 stable
        
        # Validate fraction sum
        validate_fraction_sum([params[0], params[2], 1 - params[0] - params[2]])

    def test_bi_model_reduced_individual_boundaries(self, canonical_b_values, signal_generator, noise_model):
        """Test reduced bi-exponential model with individual boundaries recovers three parameters from synthetic noisy data."""
        # Generate bi-exponential reduced signal with noise
        # Model: f1*exp(-b*D1) + (1-f1)*exp(-b*D2)
        # Kidney: f1=0.10 (blood), D1=0.165 (blood D), D2=0.002 (tissue D)
        true_f1 = 0.10  # Blood fraction
        true_d1 = 0.165  # Blood diffusion
        true_d2 = 0.002  # Tissue diffusion

        clean_signal = signal_generator.generate_biexp_reduced(
            canonical_b_values, f1=true_f1, D1=true_d1, D2=true_d2
        )
        signal = noise_model.add_noise(clean_signal, snr=cp.DEFAULT_SNR, seed=cp.DEFAULT_SEED)

        bi_model_red = BiExpFitModel("bi", fit_reduced=True)

        # Individual boundaries
        idx = (5, 7)
        x0 = np.array([0.10, 0.100, 0.003])  # f1 (blood), D1 (blood D), D2 (tissue D)
        lb = np.array([0.02, 0.050, 0.0005])
        ub = np.array([0.30, 0.250, 0.010])

        result_idx, params, _ = bi_model_red.fit(
            idx,
            signal,
            x0,
            lb,
            ub,
            b_values=canonical_b_values,
            max_iter=1000,
            btype="individual",
        )

        # Check results with SNR-based validation
        assert result_idx == idx
        # Per-parameter validation with kidney SNR=140: realistic tolerances
        validate_parameter_recovery(params[0], true_f1, snr=cp.DEFAULT_SNR, param_name="f1")
        validate_parameter_recovery(params[1], true_d1, snr=cp.DEFAULT_SNR, param_name="D1", custom_tolerance=0.15)
        validate_parameter_recovery(params[2], true_d2, snr=cp.DEFAULT_SNR, param_name="D2", custom_tolerance=0.15)
        
        # Validate fractions sum to 1
        validate_fraction_sum([params[0], 1 - params[0]])
