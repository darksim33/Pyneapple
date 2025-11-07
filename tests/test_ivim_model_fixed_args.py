"""Tests for IVIM model method behavior with fixed D components.

This module tests that the model method correctly handles fixed D parameters,
ensuring that when a D component is fixed, it's properly handled via kwargs
and the model evaluation produces correct results.
"""

import numpy as np
from pyneapple.models.ivim import (
    BiExpFitModel,
    TriExpFitModel,
)


class TestBiExpModelFixedDBehavior:
    """Test BiExpFitModel behavior with fixed D components."""

    def test_bi_exp_model_no_fixed_d(self):
        """Test BiExp model evaluation without any fixed D."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi")

        # Normal bi-exponential: f1*exp(-D1*b) + f2*exp(-D2*b)
        # Args: f1=0.3, D1=0.003, f2=0.7, D2=0.001
        result = model.model(b_values, 0.3, 0.003, 0.7, 0.001)

        # Calculate expected result manually
        expected = 0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.001)

        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert model.fix_d == 0

    def test_bi_exp_model_fixed_d1(self):
        """Test BiExp model evaluation with D1 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fix_d=1)

        # With D1=0.003 fixed (passed via kwargs), remaining args: f1=0.3, f2=0.7, D2=0.001
        result = model.model(b_values, 0.3, 0.7, 0.001, fixed_d=0.003)

        # Calculate expected result: f1*exp(-fixed_d*b) + f2*exp(-D2*b)
        expected = 0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.001)

        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert model.fix_d == 1

    def test_bi_exp_model_fixed_d2(self):
        """Test BiExp model evaluation with D2 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fix_d=2)

        # With D2=0.001 fixed (passed via kwargs), remaining args: f1=0.3, D1=0.003, f2=0.7
        result = model.model(b_values, 0.3, 0.003, 0.7, fixed_d=0.001)

        # Calculate expected result: f1*exp(-D1*b) + f2*exp(-fixed_d*b)
        expected = 0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.001)

        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert model.fix_d == 2

    def test_bi_exp_model_reduced_no_fixed_d(self):
        """Test BiExp reduced model evaluation without any fixed D."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fit_reduced=True)

        # Reduced bi-exponential: f1*exp(-D1*b) + (1-f1)*exp(-D2*b)
        # Args: f1=0.3, D1=0.003, D2=0.001
        result = model.model(b_values, 0.3, 0.003, 0.001)

        # Calculate expected result manually
        expected = 0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(
            -b_values * 0.001
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bi_exp_model_reduced_fixed_d1(self):
        """Test BiExp reduced model evaluation with D1 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fit_reduced=True, fix_d=1)

        # With D1=0.003 fixed, remaining args: f1=0.3, D2=0.001
        result = model.model(b_values, 0.3, 0.001, fixed_d=0.003)

        # Calculate expected result: f1*exp(-fixed_d*b) + (1-f1)*exp(-D2*b)
        expected = 0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(
            -b_values * 0.001
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bi_exp_model_reduced_fixed_d2(self):
        """Test BiExp reduced model evaluation with D2 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fit_reduced=True, fix_d=2)

        # With D2=0.001 fixed, remaining args: f1=0.3, D1=0.003
        result = model.model(b_values, 0.3, 0.003, fixed_d=0.001)

        # Calculate expected result: f1*exp(-D1*b) + (1-f1)*exp(-fixed_d*b)
        expected = 0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(
            -b_values * 0.001
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bi_exp_model_fit_s0_fixed_d1(self):
        """Test BiExp S0 model evaluation with D1 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fit_S0=True, fix_d=1)

        # With D1=0.003 fixed, remaining args: f1=0.3, D2=0.001, S0=1000
        result = model.model(b_values, 0.3, 0.001, 1000, fixed_d=0.003)

        # Calculate expected result: (f1*exp(-fixed_d*b) + (1-f1)*exp(-D2*b)) * S0
        expected = (
            0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(-b_values * 0.001)
        ) * 1000

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bi_exp_model_fit_s0_fixed_d2(self):
        """Test BiExp S0 model evaluation with D2 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fit_S0=True, fix_d=2)

        # With D2=0.001 fixed, remaining args: f1=0.3, D1=0.003, S0=1000
        result = model.model(b_values, 0.3, 0.003, 1000, fixed_d=0.001)

        # Calculate expected result: (f1*exp(-D1*b) + (1-f1)*exp(-fixed_d*b)) * S0
        expected = (
            0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(-b_values * 0.001)
        ) * 1000

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestTriExpModelFixedDBehavior:
    """Test TriExpFitModel behavior with fixed D components."""

    def test_tri_exp_model_no_fixed_d(self):
        """Test TriExp model evaluation without any fixed D."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri")

        # Normal tri-exponential: f1*exp(-D1*b) + f2*exp(-D2*b) + f3*exp(-D3*b)
        # Args: f1=0.2, D1=0.005, f2=0.3, D2=0.001, f3=0.5, D3=0.0002
        result = model.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.5, 0.0002)

        # Calculate expected result manually
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert model.fix_d == 0

    def test_tri_exp_model_fixed_d1(self):
        """Test TriExp model evaluation with D1 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fix_d=1)

        # With D1=0.005 fixed, remaining args: f1=0.2, f2=0.3, D2=0.001, f3=0.5, D3=0.0002
        result = model.model(b_values, 0.2, 0.3, 0.001, 0.5, 0.0002, fixed_d=0.005)

        # Calculate expected result: f1*exp(-fixed_d*b) + f2*exp(-D2*b) + f3*exp(-D3*b)
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert model.fix_d == 1

    def test_tri_exp_model_fixed_d2(self):
        """Test TriExp model evaluation with D2 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fix_d=2)

        # With D2=0.001 fixed, remaining args: f1=0.2, D1=0.005, f2=0.3, f3=0.5, D3=0.0002
        result = model.model(b_values, 0.2, 0.005, 0.3, 0.5, 0.0002, fixed_d=0.001)

        # Calculate expected result: f1*exp(-D1*b) + f2*exp(-fixed_d*b) + f3*exp(-D3*b)
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert model.fix_d == 2

    def test_tri_exp_model_fixed_d3(self):
        """Test TriExp model evaluation with D3 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fix_d=3)

        # With D3=0.0002 fixed, remaining args: f1=0.2, D1=0.005, f2=0.3, D2=0.001, f3=0.5
        result = model.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.5, fixed_d=0.0002)

        # Calculate expected result: f1*exp(-D1*b) + f2*exp(-D2*b) + f3*exp(-fixed_d*b)
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)
        assert model.fix_d == 3

    def test_tri_exp_model_reduced_fixed_d1(self):
        """Test TriExp reduced model evaluation with D1 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fit_reduced=True, fix_d=1)

        # With D1=0.005 fixed, remaining args: f1=0.2, f2=0.3, D2=0.001, D3=0.0002
        result = model.model(b_values, 0.2, 0.3, 0.001, 0.0002, fixed_d=0.005)

        # Calculate expected result: f1*exp(-fixed_d*b) + f2*exp(-D2*b) + (1-f1-f2)*exp(-D3*b)
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_reduced_fixed_d2(self):
        """Test TriExp reduced model evaluation with D2 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fit_reduced=True, fix_d=2)

        # With D2=0.001 fixed, remaining args: f1=0.2, D1=0.005, f2=0.3, D3=0.0002
        result = model.model(b_values, 0.2, 0.005, 0.3, 0.0002, fixed_d=0.001)

        # Calculate expected result: f1*exp(-D1*b) + f2*exp(-fixed_d*b) + (1-f1-f2)*exp(-D3*b)
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_reduced_fixed_d3(self):
        """Test TriExp reduced model evaluation with D3 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fit_reduced=True, fix_d=3)

        # With D3=0.0002 fixed, remaining args: f1=0.2, D1=0.005, f2=0.3, D2=0.001
        result = model.model(b_values, 0.2, 0.005, 0.3, 0.001, fixed_d=0.0002)

        # Calculate expected result: f1*exp(-D1*b) + f2*exp(-D2*b) + (1-f1-f2)*exp(-fixed_d*b)
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_fit_s0_fixed_d1(self):
        """Test TriExp S0 model evaluation with D1 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fit_S0=True, fix_d=1)

        # With D1=0.005 fixed, remaining args: f1=0.2, f2=0.3, D2=0.001, D3=0.0002, S0=1000
        result = model.model(b_values, 0.2, 0.3, 0.001, 0.0002, 1000, fixed_d=0.005)

        # Calculate expected result: (f1*exp(-fixed_d*b) + f2*exp(-D2*b) + (1-f1-f2)*exp(-D3*b)) * S0
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        ) * 1000

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_fit_s0_fixed_d3(self):
        """Test TriExp S0 model evaluation with D3 fixed."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fit_S0=True, fix_d=3)

        # With D3=0.0002 fixed, remaining args: f1=0.2, D1=0.005, f2=0.3, D2=0.001, S0=1000
        result = model.model(b_values, 0.2, 0.005, 0.3, 0.001, 1000, fixed_d=0.0002)

        # Calculate expected result: (f1*exp(-D1*b) + f2*exp(-D2*b) + (1-f1-f2)*exp(-fixed_d*b)) * S0
        expected = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        ) * 1000

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestModelFixedDEdgeCases:
    """Test edge cases and error conditions for fixed D behavior."""

    def test_fixed_d_without_kwargs(self):
        """Test that models handle missing fixed_d gracefully."""
        b_values = np.array([0, 50, 100, 200, 400, 800])

        # BiExp with fix_d=1 but no fixed_d in kwargs (should default to 0)
        model_bi = BiExpFitModel("bi", fix_d=1)
        result_bi = model_bi.model(b_values, 0.3, 0.7, 0.001)  # No fixed_d kwarg

        # Should use fixed_d=0 as default
        expected_bi = 0.3 * np.exp(-b_values * 0) + 0.7 * np.exp(-b_values * 0.001)
        np.testing.assert_allclose(result_bi, expected_bi, rtol=1e-10)

        # TriExp with fix_d=2 but no fixed_d in kwargs
        model_tri = TriExpFitModel("tri", fix_d=2)
        result_tri = model_tri.model(
            b_values, 0.2, 0.005, 0.3, 0.5, 0.0002
        )  # No fixed_d kwarg

        # Should use fixed_d=0 as default
        expected_tri = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0)
            + 0.5 * np.exp(-b_values * 0.0002)
        )
        np.testing.assert_allclose(result_tri, expected_tri, rtol=1e-10)

    def test_negative_fixed_d_values(self):
        """Test that models handle negative fixed_d values correctly (should use absolute value)."""
        b_values = np.array([0, 50, 100, 200, 400, 800])

        # BiExp with negative fixed_d value
        model_bi = BiExpFitModel("bi", fix_d=1)
        result_bi = model_bi.model(b_values, 0.3, 0.7, 0.001, fixed_d=-0.003)

        # Should use abs(fixed_d) = 0.003
        expected_bi = 0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.001)
        np.testing.assert_allclose(result_bi, expected_bi, rtol=1e-10)

    def test_zero_fixed_d_values(self):
        """Test that models handle zero fixed_d values correctly."""
        b_values = np.array([0, 50, 100, 200, 400, 800])

        # BiExp with fixed_d=0
        model_bi = BiExpFitModel("bi", fix_d=1)
        result_bi = model_bi.model(b_values, 0.3, 0.7, 0.001, fixed_d=0.0)

        # With D1=0, first component becomes f1*1 = f1
        expected_bi = 0.3 * np.ones_like(b_values) + 0.7 * np.exp(-b_values * 0.001)
        np.testing.assert_allclose(result_bi, expected_bi, rtol=1e-10)


class TestModelCompatibilityWithT1:
    """Test that fixed D behavior works correctly with T1 fitting."""

    def test_bi_exp_model_fixed_d_with_t1(self):
        """Test BiExp model with both fixed D and T1 fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fix_d=1, fit_t1=True, repetition_time=20)
        t1_value = 30
        repetition_time = 20

        # With D1=0.003 fixed and T1=30, remaining args: f1=0.3, f2=0.7, D2=0.001, T1=30
        result = model.model(b_values, 0.3, 0.7, 0.001, t1_value, fixed_d=0.003)

        # Calculate expected result with T1 term
        base_signal = 0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.001)
        expected = base_signal * (
            1 - np.exp(-repetition_time / t1_value)
        )  # Apply T1 decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_fixed_d_with_t1(self):
        """Test TriExp model with both fixed D and T1 fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fix_d=2, fit_t1=True, repetition_time=20)
        t1_value = 30
        repetition_time = 20

        # With D2=0.001 fixed and T1=30, remaining args: f1=0.2, D1=0.005, f2=0.3, f3=0.5, D3=0.0002, T1=30
        result = model.model(
            b_values, 0.2, 0.005, 0.3, 0.5, 0.0002, t1_value, fixed_d=0.001
        )

        # Calculate expected result with T1 term
        base_signal = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )
        expected = base_signal * (
            1 - np.exp(-repetition_time / t1_value)
        )  # Apply T1 decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bi_exp_model_fixed_d_with_t1_steam(self):
        """Test BiExp model with both fixed D and T1 STEAM fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel("bi", fix_d=1, fit_t1_steam=True, mixing_time=25)
        t1_value = 30
        mixing_time = 25

        # With D1=0.003 fixed and T1=30, remaining args: f1=0.3, f2=0.7, D2=0.001, T1=30
        result = model.model(b_values, 0.3, 0.7, 0.001, t1_value, fixed_d=0.003)

        # Calculate expected result with T1 STEAM term
        base_signal = 0.3 * np.exp(-b_values * 0.003) + 0.7 * np.exp(-b_values * 0.001)
        expected = base_signal * np.exp(-mixing_time / t1_value)  # Apply T1 STEAM decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_fixed_d_with_t1_steam(self):
        """Test TriExp model with both fixed D and T1 STEAM fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel("tri", fix_d=2, fit_t1_steam=True, mixing_time=25)
        t1_value = 30
        mixing_time = 25

        # With D2=0.001 fixed and T1=30, remaining args: f1=0.2, D1=0.005, f2=0.3, f3=0.5, D3=0.0002, T1=30
        result = model.model(
            b_values, 0.2, 0.005, 0.3, 0.5, 0.0002, t1_value, fixed_d=0.001
        )

        # Calculate expected result with T1 STEAM term
        base_signal = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + 0.5 * np.exp(-b_values * 0.0002)
        )
        expected = base_signal * np.exp(-mixing_time / t1_value)  # Apply T1 STEAM decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bi_exp_model_reduced_fixed_d_with_t1_steam(self):
        """Test BiExp reduced model with both fixed D and T1 STEAM fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel(
            "bi", fit_reduced=True, fix_d=1, fit_t1_steam=True, mixing_time=25
        )
        t1_value = 30
        mixing_time = 25

        # With D1=0.003 fixed and T1=30, remaining args: f1=0.3, D2=0.001, T1=30
        result = model.model(b_values, 0.3, 0.001, t1_value, fixed_d=0.003)

        # Calculate expected result with T1 STEAM term
        base_signal = 0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(
            -b_values * 0.001
        )
        expected = base_signal * np.exp(-mixing_time / t1_value)  # Apply T1 STEAM decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_reduced_fixed_d_with_t1_steam(self):
        """Test TriExp reduced model with both fixed D and T1 STEAM fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel(
            "tri", fit_reduced=True, fix_d=3, fit_t1_steam=True, mixing_time=25
        )
        t1_value = 30
        mixing_time = 25

        # With D3=0.0002 fixed and T1=30, remaining args: f1=0.2, D1=0.005, f2=0.3, D2=0.001, T1=30
        result = model.model(b_values, 0.2, 0.005, 0.3, 0.001, t1_value, fixed_d=0.0002)

        # Calculate expected result with T1 STEAM term
        base_signal = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        )
        expected = base_signal * np.exp(-mixing_time / t1_value)  # Apply T1 STEAM decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_bi_exp_model_fit_s0_fixed_d_with_t1_steam(self):
        """Test BiExp S0 model with both fixed D and T1 STEAM fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = BiExpFitModel(
            "bi", fit_S0=True, fix_d=2, fit_t1_steam=True, mixing_time=25
        )
        t1_value = 30
        mixing_time = 25

        # With D2=0.001 fixed and T1=30, remaining args: f1=0.3, D1=0.003, S0=1000, T1=30
        result = model.model(b_values, 0.3, 0.003, 1000, t1_value, fixed_d=0.001)

        # Calculate expected result with T1 STEAM term
        base_signal = (
            0.3 * np.exp(-b_values * 0.003) + (1 - 0.3) * np.exp(-b_values * 0.001)
        ) * 1000
        expected = base_signal * np.exp(-mixing_time / t1_value)  # Apply T1 STEAM decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tri_exp_model_fit_s0_fixed_d_with_t1_steam(self):
        """Test TriExp S0 model with both fixed D and T1 STEAM fitting."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        model = TriExpFitModel(
            "tri", fit_S0=True, fix_d=1, fit_t1_steam=True, mixing_time=25
        )
        t1_value = 30
        mixing_time = 25

        # With D1=0.005 fixed and T1=30, remaining args: f1=0.2, f2=0.3, D2=0.001, D3=0.0002, S0=1000, T1=30
        result = model.model(
            b_values, 0.2, 0.3, 0.001, 0.0002, 1000, t1_value, fixed_d=0.005
        )

        # Calculate expected result with T1 STEAM term
        base_signal = (
            0.2 * np.exp(-b_values * 0.005)
            + 0.3 * np.exp(-b_values * 0.001)
            + (1 - 0.2 - 0.3) * np.exp(-b_values * 0.0002)
        ) * 1000
        expected = base_signal * np.exp(-mixing_time / t1_value)  # Apply T1 STEAM decay

        np.testing.assert_allclose(result, expected, rtol=1e-10)
