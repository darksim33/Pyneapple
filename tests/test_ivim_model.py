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
    """Test IVIM model fitting with synthetic noisy data to verify parameter recovery accuracy."""

    @pytest.fixture
    def signal_mono(self, b_values):
        # Add some noise
        np.random.seed(42)
        noise = np.random.normal(0, 10, size=b_values.shape)
        return 1000 * np.exp(-b_values * 0.001) + noise

    @pytest.fixture
    def signal_bi_s0(self, b_values):
        np.random.seed(42)
        true_s0 = 1000
        true_f1 = 0.3
        true_d1 = 0.001
        true_d2 = 0.02

        signal = true_s0 * (
            true_f1 * np.exp(-b_values * true_d1)
            + (1 - true_f1) * np.exp(-b_values * true_d2)
        )
        noise = np.random.normal(0, 5, size=b_values.shape)
        return signal + noise

    def test_mono_model_fit(self, b_values, signal_mono):
        """Test that mono-exponential model fitting recovers parameters within 20% tolerance from noisy synthetic data."""
        mono_model = MonoExpFitModel("mono")
        x0 = np.array([0.002, 1000])  # Initial guess
        lb = np.array([0, 0])  # Lower bounds
        ub = np.array([0.01, 2000])  # Upper bounds

        idx, params, _ = mono_model.fit(
            0, signal_mono, x0=x0, lb=lb, ub=ub, b_values=b_values, max_iter=1000
        )

        # Check results are reasonable (within 20% of expected values)
        assert 0.0008 < params[0] < 0.0012  # D1 should be around 0.001
        assert 800 < params[1] < 1200  # S0 should be around 1000

    def test_bi_model_with_s0_fit(self, b_values, signal_bi_s0):
        """Test that bi-exponential model with S0 fitting recovers all parameters within 20% tolerance from noisy data."""
        # Create model with fit_S0=True
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)

        # Initial guess, bounds
        x0 = np.array([0.25, 0.002, 0.01, 900])  # f1, D1, D2, S0
        lb = np.array([0, 0.0007, 0.007, 500])  # Lower bounds
        ub = np.array([0.5, 0.01, 0.7, 2000])  # Upper bounds

        # Fit
        idx, params, _ = bi_model_s0.fit(
            0, signal_bi_s0, x0=x0, lb=lb, ub=ub, b_values=b_values, max_iter=1000
        )

        # Check results are reasonable (within 20% of expected values)
        assert 0.24 < params[0] < 0.36  # f1 should be around 0.3
        assert 0.0008 < params[1] < 0.0012  # D1 should be around 0.001
        assert 0.016 < params[2] < 0.024  # D2 should be around 0.02
        assert 800 < params[3] < 1200  # S0 should be around 1000

    def test_mono_model_fit_individual_boundaries(self, b_values, signal_mono):
        """Test that mono-exponential model fitting with individual boundaries recovers parameters correctly for specific pixel."""
        mono_model = MonoExpFitModel("mono")

        # Individual boundaries for specific pixel coordinate
        idx = (0, 0)
        x0 = np.array([0.002, 1000])
        lb = np.array([0, 0])
        ub = np.array([0.01, 2000])

        result_idx, params, _ = mono_model.fit(
            idx,
            signal_mono,
            x0,
            lb,
            ub,
            b_values=b_values,
            max_iter=1000,
            btype="individual",
        )

        # Check results
        assert result_idx == idx
        assert 0.0008 < params[0] < 0.0012  # D1 should be around 0.001
        assert 800 < params[1] < 1200  # S0 should be around 1000

    def test_bi_model_with_s0_fit_individual_boundaries(self, b_values, signal_bi_s0):
        """Test that bi-exponential model with S0 and individual boundaries recovers parameters correctly for specific pixel."""
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)

        # Individual boundaries for specific pixel coordinate
        idx = (1, 2)
        x0 = np.array([0.25, 0.002, 0.01, 900])
        lb = np.array([0, 0.0007, 0.007, 500])
        ub = np.array([0.5, 0.01, 0.7, 2000])

        result_idx, params, _ = bi_model_s0.fit(
            idx,
            signal_bi_s0,
            x0,
            lb,
            ub,
            b_values=b_values,
            max_iter=1000,
            btype="individual",
        )

        # Check results
        assert result_idx == idx
        assert 0.24 < params[0] < 0.36  # f1 should be around 0.3
        assert 0.0008 < params[1] < 0.0012  # D1 should be around 0.001
        assert 0.016 < params[2] < 0.024  # D2 should be around 0.02
        assert 800 < params[3] < 1200  # S0 should be around 1000

    def test_tri_model_fit_individual_boundaries(self, b_values):
        """Test that tri-exponential model with individual boundaries recovers all six parameters from synthetic noisy data."""
        # Generate tri-exponential signal with noise
        np.random.seed(42)
        true_f1 = 0.4
        true_d1 = 0.0002
        true_f2 = 0.3
        true_d2 = 0.001
        true_d3 = 0.05
        true_s0 = 1000

        signal = true_s0 * (
            true_f1 * np.exp(-b_values * true_d1)
            + true_f2 * np.exp(-b_values * true_d2)
            + (1 - true_f1 - true_f2) * np.exp(-b_values * true_d3)
        )
        noise = np.random.normal(0, 1, size=b_values.shape)
        signal += noise

        tri_model_s0 = TriExpFitModel("tri", fit_S0=True)

        # Individual boundaries
        idx = (2, 3)
        x0 = np.array([0.15, 0.0003, 0.25, 0.003, 0.02, 900])
        lb = np.array([0.05, 0.0001, 0.1, 0.0008, 0.008, 500])
        ub = np.array([0.5, 0.001, 0.5, 0.008, 0.1, 2000])

        result_idx, params, _ = tri_model_s0.fit(
            idx,
            signal,
            x0,
            lb,
            ub,
            b_values=b_values,
            max_iter=250,
            btype="individual",
        )

        # Check results
        assert result_idx == idx
        assert 0.3 < params[0] < 0.5  # f1 should be around 0.4
        assert 0.0001 < params[1] < 0.0003  # D1 should be around 0.0002
        assert 0.2 < params[2] < 0.4  # f2 should be around 0.3
        assert 0.0008 < params[3] < 0.0012  # D2 should be around 0.001
        assert 0.04 < params[4] < 0.06  # D3 should be around 0.05
        assert 800 < params[5] < 1200  # S0 should be around 1000

    def test_bi_model_reduced_individual_boundaries(self, b_values):
        """Test that reduced bi-exponential model with individual boundaries recovers three parameters from synthetic noisy data."""
        # Generate bi-exponential signal with noise
        np.random.seed(42)
        true_f1 = 0.3
        true_d1 = 0.001
        true_d2 = 0.02

        signal = true_f1 * np.exp(-b_values * true_d1) + (1 - true_f1) * np.exp(
            -b_values * true_d2
        )
        noise = np.random.normal(0, 0.01, size=b_values.shape)
        signal += noise

        bi_model_red = BiExpFitModel("bi", fit_reduced=True)

        # Individual boundaries
        idx = (5, 7)
        x0 = np.array([0.25, 0.002, 0.01])
        lb = np.array([0.1, 0.0005, 0.005])
        ub = np.array([0.5, 0.005, 0.05])

        result_idx, params, _ = bi_model_red.fit(
            idx,
            signal,
            x0,
            lb,
            ub,
            b_values=b_values,
            max_iter=1000,
            btype="individual",
        )

        # Check results
        assert result_idx == idx
        assert 0.2 < params[0] < 0.4  # f1 should be around 0.3
        assert 0.0007 < params[1] < 0.0015  # D1 should be around 0.001
        assert 0.015 < params[2] < 0.025  # D2 should be around 0.02
