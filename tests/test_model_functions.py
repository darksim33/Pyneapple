import numpy as np
import pytest
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
    apply_t1_jacobian,
)
from pyneapple.model_functions.nnls import (
    curvature_matrix,
    reconstruct_signal,
    build_regularized_basis,
    get_basis,
    get_bins,
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


class TestApplyT1Jacobian:
    """Tests for apply_t1_jacobian (standard and STEAM branches)."""

    @pytest.fixture
    def b_values(self):
        """Standard b-value array."""
        return np.array([0.0, 50.0, 100.0, 200.0, 400.0, 800.0])

    @pytest.fixture
    def dummy_jac(self, b_values):
        """Minimal 2-column Jacobian matching the b_values length."""
        n = len(b_values)
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, 2))

    @pytest.fixture
    def base_signal(self, b_values):
        """Simple monoexponential base signal."""
        return np.exp(-b_values * 0.001)

    @pytest.mark.unit
    def test_standard_t1_output_shape(self, dummy_jac, base_signal):
        """apply_t1_jacobian (no mixing_time) returns shape (n_bvalues, n_params+1)."""
        TR, T1 = 3000.0, 1000.0
        result = apply_t1_jacobian(dummy_jac, base_signal, T1, TR)
        assert result.shape == (dummy_jac.shape[0], dummy_jac.shape[1] + 1)

    @pytest.mark.unit
    def test_standard_t1_scales_columns(self, dummy_jac, base_signal):
        """Standard T1 mode scales existing columns by (1 - exp(-TR/T1))."""
        TR, T1 = 3000.0, 1000.0
        result = apply_t1_jacobian(dummy_jac, base_signal, T1, TR)
        t1_factor = 1 - np.exp(-TR / T1)
        np.testing.assert_allclose(result[:, :2], dummy_jac * t1_factor, rtol=1e-10)

    @pytest.mark.unit
    def test_standard_t1_new_column(self, dummy_jac, base_signal):
        """Standard T1 mode appends the correct dS/dT1 column."""
        TR, T1 = 3000.0, 1000.0
        result = apply_t1_jacobian(dummy_jac, base_signal, T1, TR)
        expected_jac_t1 = base_signal * (-np.exp(-TR / T1) * TR / T1**2)
        np.testing.assert_allclose(result[:, -1], expected_jac_t1, rtol=1e-10)

    @pytest.mark.unit
    def test_steam_t1_output_shape(self, dummy_jac, base_signal):
        """apply_t1_jacobian (with mixing_time) returns shape (n_bvalues, n_params+1)."""
        TR, TM, T1 = 3000.0, 50.0, 1000.0
        result = apply_t1_jacobian(dummy_jac, base_signal, T1, TR, mixing_time=TM)
        assert result.shape == (dummy_jac.shape[0], dummy_jac.shape[1] + 1)

    @pytest.mark.unit
    def test_steam_t1_scales_columns(self, dummy_jac, base_signal):
        """STEAM mode scales existing columns by (1-exp(-TR/T1)) * exp(-TM/T1)."""
        TR, TM, T1 = 3000.0, 50.0, 1000.0
        result = apply_t1_jacobian(dummy_jac, base_signal, T1, TR, mixing_time=TM)
        t1_factor = (1 - np.exp(-TR / T1)) * np.exp(-TM / T1)
        np.testing.assert_allclose(result[:, :2], dummy_jac * t1_factor, rtol=1e-10)

    @pytest.mark.unit
    def test_steam_t1_array_T1(self, dummy_jac, base_signal):
        """apply_t1_jacobian works when T1 is an array instead of a scalar."""
        TR, TM = 3000.0, 50.0
        T1 = np.full(len(base_signal), 1000.0)
        result = apply_t1_jacobian(dummy_jac, base_signal, T1, TR, mixing_time=TM)
        assert result.shape == (dummy_jac.shape[0], dummy_jac.shape[1] + 1)


class TestNNLSStandaloneFunctions:
    """Tests for curvature_matrix, reconstruct_signal, build_regularized_basis."""

    @pytest.mark.unit
    def test_curvature_matrix_shape(self):
        """curvature_matrix(n) returns a square matrix of size (n, n)."""
        n = 20
        H = curvature_matrix(n)
        assert H.shape == (n, n)

    @pytest.mark.unit
    def test_curvature_matrix_diagonal(self):
        """curvature_matrix has -2 on the main diagonal."""
        H = curvature_matrix(10)
        np.testing.assert_array_equal(np.diag(H), np.full(10, -2.0))

    @pytest.mark.unit
    def test_curvature_matrix_off_diagonal(self):
        """curvature_matrix has +1 on the first off-diagonals."""
        H = curvature_matrix(10)
        np.testing.assert_array_equal(np.diag(H, 1), np.ones(9))
        np.testing.assert_array_equal(np.diag(H, -1), np.ones(9))

    @pytest.mark.unit
    def test_reconstruct_signal_shape(self):
        """reconstruct_signal returns a 1-D array of length n_measurements."""
        bvalues = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
        d_values = get_bins(1e-4, 0.1, 30)
        rng = np.random.default_rng(0)
        spectrum = np.abs(rng.standard_normal(30))
        result = reconstruct_signal(bvalues, spectrum, d_values)
        assert result.shape == (len(bvalues),)

    @pytest.mark.unit
    def test_reconstruct_signal_matches_basis_dot(self):
        """reconstruct_signal equals get_basis(b, d) @ spectrum."""
        bvalues = np.array([0.0, 50.0, 100.0, 200.0])
        d_values = get_bins(1e-4, 0.1, 20)
        rng = np.random.default_rng(7)
        spectrum = np.abs(rng.standard_normal(20))
        expected = get_basis(bvalues, d_values) @ spectrum
        result = reconstruct_signal(bvalues, spectrum, d_values)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.unit
    def test_build_regularized_basis_shape(self):
        """build_regularized_basis returns (n_measurements + n_bins, n_bins)."""
        n_bins = 25
        bvalues = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
        d_values = get_bins(1e-4, 0.1, n_bins)
        basis = get_basis(bvalues, d_values)
        reg_basis = build_regularized_basis(basis, n_bins, order=2, mu=0.1)
        assert reg_basis.shape == (len(bvalues) + n_bins, n_bins)

    @pytest.mark.unit
    def test_build_regularized_basis_first_rows_match_basis(self):
        """First n_measurements rows of regularized basis equal the plain basis."""
        n_bins = 25
        bvalues = np.array([0.0, 50.0, 100.0, 200.0])
        d_values = get_bins(1e-4, 0.1, n_bins)
        basis = get_basis(bvalues, d_values)
        reg_basis = build_regularized_basis(basis, n_bins, order=1, mu=0.5)
        np.testing.assert_array_equal(reg_basis[: len(bvalues)], basis)
