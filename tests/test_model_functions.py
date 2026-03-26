import numpy as np
from pyneapple.model_functions.multiexp import (
    monoexp_forward,
    monoexp_reduced_forward,
    biexp_forward,
    biexp_reduced_forward,
    biexp_s0_forward,
    triexp_forward,
    triexp_reduced_forward,
    triexp_s0_forward,
    apply_t1,
    apply_t1_steam,
)


class TestMonoexpFunctions:
    def test_monoexp_forward(self):
        xdata = np.array([0, 100, 200])
        S0, D = 1000, 0.001
        expected = S0 * np.exp(-xdata * D)
        result = monoexp_forward(xdata, S0, D)
        np.testing.assert_array_almost_equal(result, expected)

    def test_monoexp_reduced_forward(self):
        xdata = np.array([0, 100, 200])
        D = 0.001
        expected = np.exp(-xdata * D)
        result = monoexp_reduced_forward(xdata, D)
        np.testing.assert_array_almost_equal(result, expected)


class TestBiexpFunctions:
    def test_biexp_forward(self):
        xdata = np.array([0, 100, 200])
        f1, D1, f2, D2 = 0.5, 0.001, 0.5, 0.0005
        expected = f1 * np.exp(-xdata * D1) + f2 * np.exp(-xdata * D2)
        result = biexp_forward(xdata, f1, D1, f2, D2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_biexp_reduced_forward(self):
        xdata = np.array([0, 100, 200])
        f1, D1, D2 = 0.5, 0.001, 0.0005
        expected = f1 * np.exp(-xdata * D1) + (1 - f1) * np.exp(-xdata * D2)
        result = biexp_reduced_forward(xdata, f1, D1, D2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_biexp_s0_forward(self):
        xdata = np.array([0, 100, 200])
        f1, D1, D2, S0 = 0.5, 0.001, 0.0005, 1000
        expected = S0 * (f1 * np.exp(-xdata * D1) + (1 - f1) * np.exp(-xdata * D2))
        result = biexp_s0_forward(xdata, f1, D1, D2, S0)
        np.testing.assert_array_almost_equal(result, expected)


class TestTriexpFunctions:
    def test_triexp_forward(self):
        xdata = np.array([0, 100, 200])
        f1, D1, f2, D2, f3, D3 = 0.3, 0.001, 0.3, 0.0005, 0.4, 0.0002
        expected = (
            f1 * np.exp(-xdata * D1)
            + f2 * np.exp(-xdata * D2)
            + f3 * np.exp(-xdata * D3)
        )
        result = triexp_forward(xdata, f1, D1, f2, D2, f3, D3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_triexp_reduced_forward(self):
        xdata = np.array([0, 100, 200])
        f1, D1, f2, D2, D3 = 0.3, 0.001, 0.3, 0.0005, 0.0002
        expected = (
            f1 * np.exp(-xdata * D1)
            + f2 * np.exp(-xdata * D2)
            + (1 - f1 - f2) * np.exp(-xdata * D3)
        )
        result = triexp_reduced_forward(xdata, f1, D1, f2, D2, D3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_triexp_s0_forward(self):
        xdata = np.array([0, 100, 200])
        f1, D1, f2, D2, D3, S0 = 0.3, 0.001, 0.3, 0.0005, 0.0002, 1000
        expected = S0 * (
            f1 * np.exp(-xdata * D1)
            + f2 * np.exp(-xdata * D2)
            + (1 - f1 - f2) * np.exp(-xdata * D3)
        )
        result = triexp_s0_forward(xdata, f1, D1, f2, D2, D3, S0)
        np.testing.assert_array_almost_equal(result, expected)


class TestT1Modifiers:
    def test_apply_t1(self):
        signal = np.array([1000, 800, 600])
        repetition_time, T1 = 3000, 1000
        expected = signal * (1 - np.exp(-repetition_time / T1))
        result = apply_t1(signal, repetition_time, T1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_t1_steam(self):
        signal = np.array([1000, 800, 600])
        mixing_time, T1 = 50, 1000
        expected = signal * np.exp(-mixing_time / T1)
        result = apply_t1_steam(signal, mixing_time, T1)
        np.testing.assert_array_almost_equal(result, expected)
