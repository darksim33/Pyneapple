import numpy as np
import pytest
from functools import partial

from pyneapple.models.ivim import MonoExpFitModel, BiExpFitModel, TriExpFitModel, get_model_class


class TestIVIMModelClasses:
    def test_mono_exp_model_creation(self):
        mono_model = MonoExpFitModel("mono")
        assert mono_model.args == ["S0", "D1"]

        # Test with T1 fitting
        mono_model_t1 = MonoExpFitModel("mono", mixing_time=20, fit_t1=True)
        assert mono_model_t1.args == ["S0", "D1", "T1"]

        # Test fit_reduced model
        mono_model_reduced = MonoExpFitModel("mono", fit_reduced=True)
        assert mono_model_reduced.args == ["D1"]

    def test_bi_exp_model_creation(self):
        bi_model = BiExpFitModel("bi")
        assert bi_model.args == ["f1", "D1", "f2", "D2"]

        # Test with fit_reduced model
        bi_model_reduced = BiExpFitModel("bi", fit_reduced=True)
        assert bi_model_reduced.args == ["f1", "D1", "D2"]

        # Test with T1 fitting
        bi_model_t1 = BiExpFitModel("bi", mixing_time=20, fit_t1=True)
        assert bi_model_t1.args == ["f1", "D1", "f2", "D2", "T1"]

    def test_bi_exp_model_fit_s0_creation(self):
        # Test initialization with fit_S0=True
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)
        assert bi_model_s0.args == ["f1", "D1", "D2", "S0"]
        assert bi_model_s0.fit_S0 is True

        # Test with T1 fitting
        bi_model_s0_t1 = BiExpFitModel("bi", fit_S0=True, mixing_time=20, fit_t1=True)
        assert bi_model_s0_t1.args == ["f1", "D1", "D2", "S0", "T1"]
        assert bi_model_s0_t1.fit_S0 is True

        # Test with fit_reduced model (should raise ValueError)
        with pytest.raises(ValueError):
            BiExpFitModel("bi", fit_S0=True, fit_reduced=True)

    def test_tri_exp_model_creation(self):
        tri_model = TriExpFitModel("tri")
        assert tri_model.args == ["f1", "D1", "f2", "D2", "f3", "D3"]

        # Test with fit_reduced model
        tri_model_reduced = TriExpFitModel("tri", fit_reduced=True)
        assert tri_model_reduced.args == ["f1", "D1", "f2", "D2", "D3"]

        # Test with T1 fitting
        tri_model_t1 = TriExpFitModel("tri", mixing_time=20, fit_t1=True)
        assert tri_model_t1.args == ["f1", "D1", "f2", "D2", "f3", "D3", "T1"]

    def test_tri_exp_model_fit_s0_creation(self):
        # Test initialization with fit_S0=True
        tri_model_s0 = TriExpFitModel("tri", fit_S0=True)
        assert tri_model_s0.args == ["f1", "D1", "f2", "D2", "D3", "S0"]
        assert tri_model_s0.fit_S0 is True

        # Test with T1 fitting
        tri_model_s0_t1 = TriExpFitModel("tri", fit_S0=True, mixing_time=20, fit_t1=True)
        assert tri_model_s0_t1.args == ["f1", "D1", "f2", "D2", "D3", "S0", "T1"]
        assert tri_model_s0_t1.fit_S0 is True

        # Test with fit_reduced model (should raise ValueError)
        with pytest.raises(ValueError):
            TriExpFitModel("tri", fit_S0=True, fit_reduced=True)

    def test_get_model_class(self):
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
    @pytest.fixture
    def signal_mono(self, b_values):
        # S0=1000, D1=0.001
        return 1000 * np.exp(-b_values * 0.001)

    @pytest.fixture
    def signal_bi(self, b_values):
        # f1=0.3, D1=0.003, f2=0.7 D2=0.0005
        return (0.3 * np.exp(-b_values * 0.003) +
                0.7 * np.exp(-b_values * 0.0005))

    @pytest.fixture
    def signal_bi_s0(self, b_values):
        # f1=0.3, D1=0.003, f2=0.7 D2=0.0005, S0=1000
        return (0.3 * np.exp(-b_values * 0.003) +
                0.7 * np.exp(-b_values * 0.0005)) * 1000

    @pytest.fixture
    def signal_tri(self, b_values):
        # f1=0.2, D1=0.005, f2=0.3, D2=0.001, f3=0.5, D3=0.0002
        return (0.2 * np.exp(-b_values * 0.005) +
                0.3 * np.exp(-b_values * 0.001) +
                0.5 * np.exp(-b_values * 0.0002))

    @pytest.fixture
    def signal_tri_s0(self, b_values):
        # f1=0.2, D1=0.005, f2=0.3, D2=0.001, D3=0.0002, S0=1000
        return 1000 * (0.2 * np.exp(-b_values * 0.005) +
                       0.3 * np.exp(-b_values * 0.001) +
                       0.5 * np.exp(-b_values * 0.0002))

    def test_mono_model_evaluation(self, b_values, signal_mono):
        mono_model = MonoExpFitModel("mono")
        # Test with correct parameters
        output = mono_model.model(b_values, 1000, 0.001)
        np.testing.assert_allclose(output, signal_mono, rtol=1e-5)

    def test_bi_model_evaluation(self, b_values, signal_bi):
        bi_model = BiExpFitModel("bi")
        # Test with correct parameters
        output = bi_model.model(b_values, 0.3, 0.003, 0.7, 0.0005)
        np.testing.assert_allclose(output, signal_bi, rtol=1e-5)

        # Test fit_reduced model
        bi_model_red = BiExpFitModel("bi", fit_reduced=True)
        output_red = bi_model_red.model(b_values, 0.3, 0.003, 0.0005)
        np.testing.assert_allclose(output_red, signal_bi, rtol=1e-5)

        # Test fixed D
        bi_model_fixed = BiExpFitModel("bi", fix_d=True)
        output_fixed = bi_model_fixed.model(b_values, 0.3, 0.003, 0.7, 0, fixed_d=0.0005)
        np.testing.assert_allclose(output_fixed, signal_bi, rtol=1e-5)

    def test_bi_model_with_s0_evaluation(self, b_values, signal_bi_s0):
        # Test model with fit_S0=True
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)
        output = bi_model_s0.model(b_values, 0.3, 0.003, 0.0005, 1000)
        np.testing.assert_allclose(output, signal_bi_s0, rtol=1e-5)

        # Test with T1 fitting
        mixing_time = 20
        t1_value = 30
        signal_bi_s0_t1 = signal_bi_s0 * np.exp(-t1_value / mixing_time)

        bi_model_s0_t1 = BiExpFitModel("bi", fit_S0=True, mixing_time=mixing_time, fit_t1=True)
        output_t1 = bi_model_s0_t1.model(b_values, 0.3, 0.003, 0.0005, 1000, t1_value)
        np.testing.assert_allclose(output_t1, signal_bi_s0_t1, rtol=1e-5)

    def test_tri_model_evaluation(self, b_values, signal_tri):
        tri_model = TriExpFitModel("tri")
        # Test with correct parameters
        output = tri_model.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.5, 0.0002)
        np.testing.assert_allclose(output, signal_tri, rtol=1e-5)

        # Test fit_reduced model
        tri_model_red = TriExpFitModel("tri", fit_reduced=True)
        output_red = tri_model_red.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.0002)
        np.testing.assert_allclose(output_red, signal_tri, rtol=1e-5)

        # Test fixed D
        tri_model_fixed = TriExpFitModel("tri", fix_d=True)
        output_fixed = tri_model_fixed.model(
            b_values, 0.2, 0.005, 0.3, 0.001, 0.5, 0, fixed_d=0.0002
        )
        np.testing.assert_allclose(output_fixed, signal_tri, rtol=1e-5)

    def test_tri_model_with_s0_evaluation(self, b_values, signal_tri_s0):
        # Test model with fit_S0=True
        tri_model_s0 = TriExpFitModel("tri", fit_S0=True)
        output = tri_model_s0.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.0002, 1000)
        np.testing.assert_allclose(output, signal_tri_s0, rtol=1e-5)

        # Test with T1 fitting
        mixing_time = 20
        t1_value = 30
        signal_tri_s0_t1 = signal_tri_s0 * np.exp(-t1_value / mixing_time)

        tri_model_s0_t1 = TriExpFitModel("tri", fit_S0=True, mixing_time=mixing_time, fit_t1=True)
        output_t1 = tri_model_s0_t1.model(b_values, 0.2, 0.005, 0.3, 0.001, 0.0002, 1000, t1_value)
        np.testing.assert_allclose(output_t1, signal_tri_s0_t1, rtol=1e-5)


class TestIVIMModelFitting:

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

        signal = true_s0 * (true_f1 * np.exp(-b_values * true_d1) +
                            (1 - true_f1) * np.exp(-b_values * true_d2))
        noise = np.random.normal(0, 5, size=b_values.shape)
        return signal + noise

    def test_mono_model_fit(self, b_values, signal_mono):
        mono_model = MonoExpFitModel("mono")
        x0 = np.array([1000, 0.002])  # Initial guess
        lb = np.array([0, 0])  # Lower bounds
        ub = np.array([2000, 0.01])  # Upper bounds

        idx, params, _ = mono_model.fit(
            0, signal_mono, x0=x0, lb=lb, ub=ub,
            b_values=b_values, max_iter=1000
        )

        # Check results are reasonable (within 20% of expected values)
        assert 800 < params[0] < 1200  # S0 should be around 1000
        assert 0.0008 < params[1] < 0.0012  # D1 should be around 0.001

    def test_bi_model_with_s0_fit(self, b_values, signal_bi_s0):
        # Create model with fit_S0=True
        bi_model_s0 = BiExpFitModel("bi", fit_S0=True)

        # Initial guess, bounds
        x0 = np.array([0.25, 0.002, 0.01, 900])  # f1, D1, D2, S0
        lb = np.array([0, 0.0007, 0.007, 500])  # Lower bounds
        ub = np.array([0.5, 0.01, 0.7, 2000])  # Upper bounds

        # Fit
        idx, params, _ = bi_model_s0.fit(
            0, signal_bi_s0, x0=x0, lb=lb, ub=ub,
            b_values=b_values, max_iter=1000
        )

        # Check results are reasonable (within 20% of expected values)
        assert 0.24 < params[0] < 0.36  # f1 should be around 0.3
        assert 0.0008 < params[1] < 0.0012  # D1 should be around 0.001
        assert 0.016 < params[2] < 0.024  # D2 should be around 0.02
        assert 800 < params[3] < 1200  # S0 should be around 1000
