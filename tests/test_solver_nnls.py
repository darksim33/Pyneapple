"""Tests for NNLSSolver — initialization, basis construction, regularization, and fitting."""

import numpy as np
import pytest

from pyneapple.models import NNLSModel
from pyneapple.solvers.base import _PixelFitResult
from pyneapple.solvers.nnls_solver import NNLSSolver


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=float)
D_RANGE = (1e-4, 0.1)
N_BINS = 50  # Small for fast tests


@pytest.fixture
def b_values():
    """Standard b-value array."""
    return B_VALUES.copy()


@pytest.fixture
def d_range():
    """Diffusion coefficient range (d_min, d_max)."""
    return D_RANGE


@pytest.fixture
def n_bins():
    """Number of diffusion coefficient bins for testing."""
    return N_BINS


@pytest.fixture
def nnls_solver(d_range, n_bins):
    """NNLSSolver with no regularization (reg_order=0)."""
    return NNLSSolver(
        model=NNLSModel(d_range=d_range, n_bins=n_bins),
        reg_order=0,
        mu=0.02,
        max_iter=250,
        tol=1e-8,
    )


@pytest.fixture
def synthetic_single(b_values):
    """Noise-free mono-exponential signal for a single voxel."""
    D_true = 0.001  # within D_RANGE
    S0_true = 1000.0
    signal = S0_true * np.exp(-b_values * D_true)
    return signal, S0_true, D_true


@pytest.fixture
def synthetic_multi(b_values):
    """Noise-free mono-exponential signals for 3 voxels with different parameters."""
    param_sets = [
        (1000.0, 0.001),
        (800.0, 0.002),
        (500.0, 0.0005),
    ]
    signals = np.array(
        [S0 * np.exp(-b_values * D) for S0, D in param_sets]
    )  # shape (3, 8)
    return signals, param_sets


# ---------------------------------------------------------------------------
# TestNNLSSolverInit
# ---------------------------------------------------------------------------


class TestNNLSSolverInit:
    """Test suite for NNLSSolver construction and attribute validation."""

    def test_valid_initialization(self, d_range, n_bins):
        """NNLSSolver initializes correctly with all valid parameters."""
        model = NNLSModel(d_range=d_range, n_bins=n_bins)
        solver = NNLSSolver(
            model=model,
            reg_order=0,
            mu=0.02,
            max_iter=300,
            tol=1e-7,
        )
        assert solver.model.d_range == d_range
        assert solver.model.n_bins == n_bins
        assert solver.reg_order == 0
        assert solver.mu == pytest.approx(0.02)
        assert solver.max_iter == 300
        assert solver.tol == pytest.approx(1e-7)

    def test_model_stored_on_solver(self, d_range, n_bins):
        """NNLSSolver stores the provided NNLSModel instance."""
        model = NNLSModel(d_range=d_range, n_bins=n_bins)
        solver = NNLSSolver(model=model)
        assert solver.model is model

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ("reg_order", 0),
            ("mu", pytest.approx(0.02)),
            ("max_iter", 250),
            ("multi_threading", False),
        ],
    )
    def test_nnls_solver_defaults(self, d_range, n_bins, attr, expected):
        """NNLSSolver has correct default values."""
        solver = NNLSSolver(model=NNLSModel(d_range=d_range, n_bins=n_bins))
        assert getattr(solver, attr) == expected

    def test_multithreading_can_be_enabled(self, d_range, n_bins):
        """Multi-threading can be enabled via constructor argument."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), multi_threading=True
        )
        assert solver.multi_threading is True

    def test_params_and_diagnostics_empty_before_fit(self, nnls_solver):
        """params_ and diagnostics_ are empty dicts before fit() is called."""
        assert len(nnls_solver.params_) == 0
        assert len(nnls_solver.diagnostics_) == 0


# ---------------------------------------------------------------------------
# TestNNLSSolverBeforeFit
# ---------------------------------------------------------------------------


class TestNNLSSolverBeforeFit:
    """Tests for BaseSolver helpers called before fit()."""

    def test_get_params_raises_before_fit(self, nnls_solver):
        """get_params() raises RuntimeError when called before fit()."""
        with pytest.raises(RuntimeError, match="No parameters available"):
            nnls_solver.get_params()

    def test_get_diagnostics_raises_before_fit(self, nnls_solver):
        """get_diagnostics() raises RuntimeError when called before fit()."""
        with pytest.raises(RuntimeError, match="No diagnostics available"):
            nnls_solver.get_diagnostics()


# ---------------------------------------------------------------------------
# TestNNLSSolverBins
# ---------------------------------------------------------------------------


class TestNNLSSolverBins:
    """Tests for the NNLSModel.bins property (accessed via solver.model)."""

    def test_bins_shape(self, nnls_solver, n_bins):
        """bins returns a 1D array of length n_bins."""
        assert nnls_solver.model.bins.shape == (n_bins,)

    def test_bins_start_at_d_min(self, nnls_solver, d_range):
        """First bin equals d_min of d_range."""
        assert nnls_solver.model.bins[0] == pytest.approx(d_range[0])

    def test_bins_end_at_d_max(self, nnls_solver, d_range):
        """Last bin equals d_max of d_range."""
        assert nnls_solver.model.bins[-1] == pytest.approx(d_range[1])

    def test_bins_are_monotonically_increasing(self, nnls_solver):
        """bins are strictly monotonically increasing."""
        assert np.all(np.diff(nnls_solver.model.bins) > 0), (
            "bins should be strictly increasing"
        )

    def test_bins_are_all_positive(self, nnls_solver):
        """All bin values are strictly positive."""
        assert np.all(nnls_solver.model.bins > 0), "All diffusion bins must be positive"

    def test_bins_are_log_spaced(self, nnls_solver):
        """Consecutive bin ratios are constant, confirming logarithmic spacing."""
        bins = nnls_solver.model.bins
        log_bins = np.log10(bins)
        diffs = np.diff(log_bins)
        np.testing.assert_allclose(
            diffs, diffs[0], rtol=1e-10, err_msg="bins are not log-spaced"
        )

    def test_bins_dtype_is_float(self, nnls_solver):
        """bins array dtype is floating-point."""
        assert np.issubdtype(nnls_solver.model.bins.dtype, np.floating)


# ---------------------------------------------------------------------------
# TestNNLSSolverGetBasis
# ---------------------------------------------------------------------------


class TestNNLSSolverGetBasis:
    """Tests for NNLSModel.get_basis() (accessed via solver.model)."""

    def test_basis_shape(self, nnls_solver, b_values, n_bins):
        """get_basis returns matrix of shape (n_measurements, n_bins)."""
        basis = nnls_solver.model.get_basis(b_values)
        assert basis.shape == (len(b_values), n_bins)

    def test_basis_at_b0_is_all_ones(self, nnls_solver, n_bins):
        """At b=0, all basis values equal 1.0 (exp(-0 * d) = 1)."""
        basis = nnls_solver.model.get_basis(np.array([0.0]))
        np.testing.assert_allclose(basis, np.ones((1, n_bins)))

    def test_basis_values_are_positive(self, nnls_solver, b_values):
        """All basis values are strictly positive (exponentials are always positive)."""
        basis = nnls_solver.model.get_basis(b_values)
        assert np.all(basis > 0), "Basis values must be positive"

    def test_basis_values_are_at_most_one(self, nnls_solver, b_values):
        """All basis values are at most 1.0 for non-negative b-values."""
        basis = nnls_solver.model.get_basis(b_values)
        assert np.all(basis <= 1.0 + 1e-10), "Basis values must not exceed 1.0"

    def test_basis_decays_monotonically_with_b(self, nnls_solver, n_bins):
        """For every diffusion bin column, signal decays monotonically as b increases."""
        b = np.array([0.0, 100.0, 500.0, 1000.0])
        basis = nnls_solver.model.get_basis(b)
        for j in range(n_bins):
            assert np.all(np.diff(basis[:, j]) <= 0), (
                f"Column {j} should be non-increasing with b-value"
            )

    def test_basis_raises_for_2d_xdata(self, nnls_solver):
        """get_basis raises ValueError when xdata is not 1D."""
        bad_xdata = np.ones((3, 4))
        with pytest.raises(ValueError):
            nnls_solver.model.get_basis(bad_xdata)

    def test_basis_higher_d_decays_faster(self, nnls_solver, b_values):
        """Columns with larger D values decay faster (larger absolute signal drop)."""
        basis = nnls_solver.model.get_basis(b_values)
        # Compare first and last column (d_min vs d_max)
        drop_first = basis[0, 0] - basis[-1, 0]  # slow diffusion (small D)
        drop_last = basis[0, -1] - basis[-1, -1]  # fast diffusion (large D)
        assert drop_last > drop_first, "Larger D bins should show faster signal decay"


# ---------------------------------------------------------------------------
# TestNNLSSolverRegularization
# ---------------------------------------------------------------------------


class TestNNLSSolverRegularization:
    """Tests for NNLSSolver.get_regularization_matrix()."""

    def test_reg_order_0_returns_zero_matrix(self, nnls_solver, n_bins):
        """reg_order=0 returns an all-zero matrix (no regularization)."""
        reg = nnls_solver.get_regularization_matrix()
        assert reg.shape == (n_bins, n_bins)
        np.testing.assert_array_equal(reg, np.zeros((n_bins, n_bins)))

    @pytest.mark.parametrize("reg_order", [1, 2, 3], ids=["order1", "order2", "order3"])
    def test_reg_matrix_shape(self, d_range, n_bins, reg_order):
        """Regularization matrix has shape (n_bins, n_bins) for all supported orders."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=reg_order, mu=1.0
        )
        reg = solver.get_regularization_matrix()
        assert reg.shape == (n_bins, n_bins)

    @pytest.mark.parametrize("reg_order", [1, 2, 3], ids=["order1", "order2", "order3"])
    def test_reg_matrix_scales_linearly_with_mu(self, d_range, n_bins, reg_order):
        """Regularization matrix entries scale linearly with mu."""
        solver_1 = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=reg_order, mu=1.0
        )
        solver_2 = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=reg_order, mu=2.0
        )
        reg_1 = solver_1.get_regularization_matrix()
        reg_2 = solver_2.get_regularization_matrix()
        np.testing.assert_allclose(reg_2, 2.0 * reg_1, rtol=1e-10)

    def test_reg_order_1_diagonal_is_negative_one(self, d_range, n_bins):
        """reg_order=1 main diagonal is -1 * mu (first-difference matrix)."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=1, mu=1.0
        )
        reg = solver.get_regularization_matrix()
        assert reg[0, 0] == pytest.approx(-1.0)
        assert reg[0, 1] == pytest.approx(1.0)

    def test_reg_order_2_center_diagonal_is_negative_two(self, d_range, n_bins):
        """reg_order=2 center entry is -2 * mu and neighbors are +1 * mu."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=2, mu=1.0
        )
        reg = solver.get_regularization_matrix()
        assert reg[1, 0] == pytest.approx(1.0)
        assert reg[1, 1] == pytest.approx(-2.0)
        assert reg[1, 2] == pytest.approx(1.0)

    def test_reg_order_3_center_diagonal_is_negative_six(self, d_range, n_bins):
        """reg_order=3 center diagonal entry is -6 * mu."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=3, mu=1.0
        )
        reg = solver.get_regularization_matrix()
        assert reg[2, 2] == pytest.approx(-6.0)

    def test_unsupported_reg_order_raises(self, d_range, n_bins):
        """Unsupported regularization order raises NotImplementedError."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=99
        )
        with pytest.raises(NotImplementedError, match="Regularization order"):
            solver.get_regularization_matrix()


# ---------------------------------------------------------------------------
# TestNNLSSolverBuildRegularizedBasis
# ---------------------------------------------------------------------------


class TestNNLSSolverBuildRegularizedBasis:
    """Tests for NNLSSolver._build_regularized_basis()."""

    def test_shape_no_regularization(self, nnls_solver, b_values, n_bins):
        """_build_regularized_basis returns shape (n_measurements + n_bins, n_bins)."""
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        assert reg_basis.shape == (len(b_values) + n_bins, n_bins)

    @pytest.mark.parametrize(
        "reg_order", [0, 1, 2, 3], ids=["order0", "order1", "order2", "order3"]
    )
    def test_shape_for_all_reg_orders(self, d_range, n_bins, b_values, reg_order):
        """_build_regularized_basis returns correct shape for all supported reg_orders."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=reg_order
        )
        reg_basis = solver._build_regularized_basis(b_values)
        assert reg_basis.shape == (len(b_values) + n_bins, n_bins)

    def test_top_rows_match_get_basis(self, nnls_solver, b_values):
        """First n_measurements rows of _build_regularized_basis match get_basis() output."""
        basis = nnls_solver.model.get_basis(b_values)
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        np.testing.assert_array_equal(reg_basis[: len(b_values)], basis)

    def test_bottom_rows_match_reg_matrix(self, nnls_solver, b_values):
        """Last n_bins rows of _build_regularized_basis match get_regularization_matrix() output."""
        reg_matrix = nnls_solver.get_regularization_matrix()
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        np.testing.assert_array_equal(reg_basis[len(b_values) :], reg_matrix)


# ---------------------------------------------------------------------------
# TestNNLSSolverExtendSignal
# ---------------------------------------------------------------------------


class TestNNLSSolverExtendSignal:
    """Tests for NNLSSolver._extend_signal()."""

    def test_output_shape_single_pixel(self, nnls_solver, b_values, n_bins):
        """_extend_signal returns (1, n_measurements + n_bins) for a single pixel."""
        signal = np.ones((1, len(b_values)))
        extended = nnls_solver._extend_signal(signal)
        assert extended.shape == (1, len(b_values) + n_bins)

    def test_output_shape_multi_pixel(self, nnls_solver, b_values, n_bins):
        """_extend_signal returns (n_pixels, n_measurements + n_bins) for multiple pixels."""
        n_pixels = 5
        signal = np.ones((n_pixels, len(b_values)))
        extended = nnls_solver._extend_signal(signal)
        assert extended.shape == (n_pixels, len(b_values) + n_bins)

    def test_original_signal_preserved(self, nnls_solver, b_values):
        """_extend_signal preserves the original signal values in the leading columns."""
        rng = np.random.default_rng(42)
        signal = rng.random((3, len(b_values)))
        extended = nnls_solver._extend_signal(signal)
        np.testing.assert_array_equal(extended[:, : len(b_values)], signal)

    def test_padding_is_zeros(self, nnls_solver, b_values, n_bins):
        """_extend_signal appends n_bins zero columns (regularization padding)."""
        signal = np.ones((2, len(b_values)))
        extended = nnls_solver._extend_signal(signal)
        np.testing.assert_array_equal(
            extended[:, len(b_values) :], np.zeros((2, n_bins))
        )


# ---------------------------------------------------------------------------
# TestNNLSSolverFitSinglePixel
# ---------------------------------------------------------------------------


class TestNNLSSolverFitSinglePixel:
    """Tests for NNLSSolver._fit_single_pixel()."""

    def test_returns_tuple_of_two_elements(self, nnls_solver, b_values):
        """_fit_single_pixel returns a _PixelFitResult with params and residual."""
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        signal = np.ones(reg_basis.shape[0])
        result = nnls_solver._fit_single_pixel(reg_basis, signal, pixel_idx=0)
        assert isinstance(result, _PixelFitResult)

    def test_coefficients_shape(self, nnls_solver, b_values, n_bins):
        """_fit_single_pixel returns coefficients of shape (n_bins,)."""
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        signal = np.ones(reg_basis.shape[0])
        pr = nnls_solver._fit_single_pixel(reg_basis, signal, pixel_idx=0)
        assert pr.params.shape == (n_bins,)

    def test_coefficients_are_non_negative(self, nnls_solver, b_values):
        """_fit_single_pixel returns non-negative coefficients (NNLS hard constraint)."""
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        signal = np.ones(reg_basis.shape[0])
        pr = nnls_solver._fit_single_pixel(reg_basis, signal, pixel_idx=0)
        assert np.all(pr.params >= 0), "NNLS coefficients must be non-negative"

    def test_residual_is_scalar(self, nnls_solver, b_values):
        """_fit_single_pixel returns a scalar (0-d) residual value."""
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        signal = np.ones(reg_basis.shape[0])
        pr = nnls_solver._fit_single_pixel(reg_basis, signal, pixel_idx=0)
        assert np.ndim(pr.residual) == 0 or np.isscalar(pr.residual)

    def test_residual_is_non_negative(self, nnls_solver, b_values, synthetic_single):
        """NNLS residual is always non-negative."""
        signal_1d, _, _ = synthetic_single
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        extended = np.concatenate([signal_1d, np.zeros(nnls_solver.model.n_bins)])
        pr = nnls_solver._fit_single_pixel(reg_basis, extended, pixel_idx=0)
        assert pr.residual >= 0.0, (
            f"NNLS residual must be non-negative, got {pr.residual}"
        )

    def test_failed_fit_returns_zeros_and_norm_residual(
        self, nnls_solver, b_values, mocker
    ):
        """When nnls raises an exception, _fit_single_pixel returns zero coefficients
        and the L2 norm of the signal as residual (equivalent to all-zero coefficient fit)."""
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        signal = np.ones(reg_basis.shape[0])
        mocker.patch(
            "pyneapple.solvers.nnls_solver.nnls",
            side_effect=RuntimeError("NNLS failed"),
        )
        pr = nnls_solver._fit_single_pixel(reg_basis, signal, pixel_idx=0)
        np.testing.assert_array_equal(pr.params, np.zeros(nnls_solver.model.n_bins))
        assert pr.residual == pytest.approx(float(np.linalg.norm(signal)))


# ---------------------------------------------------------------------------
# TestNNLSSolverFit
# ---------------------------------------------------------------------------


class TestNNLSSolverFit:
    """End-to-end tests for NNLSSolver.fit()."""

    @pytest.mark.unit
    def test_fit_returns_self(self, nnls_solver, b_values, synthetic_single):
        """fit() returns the solver instance itself to enable method chaining."""
        signal, _, _ = synthetic_single
        result = nnls_solver.fit(b_values, signal)
        assert result is nnls_solver

    @pytest.mark.unit
    def test_fit_stores_coefficients_in_params(
        self, nnls_solver, b_values, synthetic_single
    ):
        """After fit(), params_['coefficients'] is populated."""
        signal, _, _ = synthetic_single
        nnls_solver.fit(b_values, signal)
        assert "coefficients" in nnls_solver.params_

    @pytest.mark.unit
    def test_fit_stores_residual_in_diagnostics(
        self, nnls_solver, b_values, synthetic_single
    ):
        """After fit(), diagnostics_['residual'] is populated."""
        signal, _, _ = synthetic_single
        nnls_solver.fit(b_values, signal)
        assert "residual" in nnls_solver.diagnostics_

    @pytest.mark.unit
    def test_fit_1d_signal_produces_coefficients_shape_1_by_nbins(
        self, nnls_solver, b_values, synthetic_single, n_bins
    ):
        """fit() with a 1D signal produces coefficients of shape (1, n_bins)."""
        signal, _, _ = synthetic_single
        assert signal.ndim == 1, "Fixture signal should be 1D"
        nnls_solver.fit(b_values, signal)
        assert nnls_solver.params_["coefficients"].shape == (1, n_bins)

    @pytest.mark.unit
    def test_fit_multi_voxel_coefficients_shape(
        self, nnls_solver, b_values, synthetic_multi, n_bins
    ):
        """After multi-voxel fit, coefficients have shape (n_pixels, n_bins)."""
        signals, param_sets = synthetic_multi
        nnls_solver.fit(b_values, signals)
        assert nnls_solver.params_["coefficients"].shape == (len(param_sets), n_bins)

    @pytest.mark.unit
    def test_fit_coefficients_are_non_negative(
        self, nnls_solver, b_values, synthetic_multi
    ):
        """All fitted coefficients are non-negative (NNLS constraint)."""
        signals, _ = synthetic_multi
        nnls_solver.fit(b_values, signals)
        coeffs = nnls_solver.params_["coefficients"]
        assert np.all(coeffs >= 0), "All NNLS coefficients must be non-negative"

    @pytest.mark.unit
    def test_fit_residuals_shape_single_voxel(
        self, nnls_solver, b_values, synthetic_single
    ):
        """After single-voxel fit, diagnostics residual array has shape (1,)."""
        signal, _, _ = synthetic_single
        nnls_solver.fit(b_values, signal)
        assert nnls_solver.diagnostics_["residual"].shape == (1,)

    @pytest.mark.unit
    def test_fit_residuals_shape_multi_voxel(
        self, nnls_solver, b_values, synthetic_multi
    ):
        """After multi-voxel fit, diagnostics residual array has shape (n_pixels,)."""
        signals, param_sets = synthetic_multi
        nnls_solver.fit(b_values, signals)
        assert nnls_solver.diagnostics_["residual"].shape == (len(param_sets),)

    @pytest.mark.unit
    def test_fit_resets_state_on_second_call(
        self, nnls_solver, b_values, synthetic_single
    ):
        """Calling fit() a second time overwrites previous coefficients cleanly."""
        signal, _, _ = synthetic_single
        nnls_solver.fit(b_values, signal)
        first_coeffs = nnls_solver.params_["coefficients"].copy()
        nnls_solver.fit(b_values, signal * 0.5)  # different amplitude
        second_coeffs = nnls_solver.params_["coefficients"]
        assert not np.allclose(first_coeffs, second_coeffs), (
            "Coefficients should differ after re-fitting on different data"
        )

    @pytest.mark.unit
    def test_fit_recovers_peak_near_true_diffusivity(
        self, nnls_solver, b_values, synthetic_single
    ):
        """For a mono-exponential signal, the peak coefficient bin is near the true D."""
        signal, _, D_true = synthetic_single
        nnls_solver.fit(b_values, signal)
        coeffs = nnls_solver.params_["coefficients"][0]
        peak_d = nnls_solver.model.bins[np.argmax(coeffs)]
        # Allow ±1 decade tolerance due to discrete bin approximation
        assert abs(np.log10(peak_d) - np.log10(D_true)) < 1.0, (
            f"Peak bin D={peak_d:.4e} is too far from true D={D_true:.4e}"
        )

    @pytest.mark.unit
    def test_get_params_after_fit_contains_coefficients(
        self, nnls_solver, b_values, synthetic_single
    ):
        """get_params() returns a dict containing 'coefficients' after fit()."""
        signal, _, _ = synthetic_single
        nnls_solver.fit(b_values, signal)
        params = nnls_solver.get_params()
        assert isinstance(params, dict)
        assert "coefficients" in params

    @pytest.mark.unit
    def test_get_diagnostics_after_fit_contains_residual(
        self, nnls_solver, b_values, synthetic_single
    ):
        """get_diagnostics() returns a dict containing 'residual' after fit()."""
        signal, _, _ = synthetic_single
        nnls_solver.fit(b_values, signal)
        diag = nnls_solver.get_diagnostics()
        assert isinstance(diag, dict)
        assert "residual" in diag


# ---------------------------------------------------------------------------
# TestNNLSSolverFitData
# ---------------------------------------------------------------------------


class TestNNLSSolverFitData:
    """Tests for NNLSSolver._fit_data() output shapes and content."""

    def test_coefficients_shape_single_pixel(
        self, nnls_solver, b_values, synthetic_single, n_bins
    ):
        """_fit_data returns coefficients of shape (1, n_bins) for a single pixel."""
        signal, _, _ = synthetic_single
        signal_2d = signal[np.newaxis, :]
        nnls_solver.n_pixels = 1
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        extended = nnls_solver._extend_signal(signal_2d)
        coeffs, _ = nnls_solver._fit_data(reg_basis, extended)
        assert coeffs.shape == (1, n_bins)

    def test_residuals_shape_single_pixel(
        self, nnls_solver, b_values, synthetic_single
    ):
        """_fit_data returns residuals of shape (1,) for a single pixel."""
        signal, _, _ = synthetic_single
        signal_2d = signal[np.newaxis, :]
        nnls_solver.n_pixels = 1
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        extended = nnls_solver._extend_signal(signal_2d)
        _, residuals = nnls_solver._fit_data(reg_basis, extended)
        assert residuals.shape == (1,)

    def test_coefficients_shape_multi_pixel(
        self, nnls_solver, b_values, synthetic_multi, n_bins
    ):
        """_fit_data returns coefficients of shape (n_pixels, n_bins) for multiple pixels."""
        signals, param_sets = synthetic_multi
        n_pixels = len(param_sets)
        nnls_solver.n_pixels = n_pixels
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        extended = nnls_solver._extend_signal(signals)
        coeffs, _ = nnls_solver._fit_data(reg_basis, extended)
        assert coeffs.shape == (n_pixels, n_bins)

    def test_residuals_shape_multi_pixel(self, nnls_solver, b_values, synthetic_multi):
        """_fit_data returns residuals of shape (n_pixels,) for multiple pixels."""
        signals, param_sets = synthetic_multi
        n_pixels = len(param_sets)
        nnls_solver.n_pixels = n_pixels
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        extended = nnls_solver._extend_signal(signals)
        _, residuals = nnls_solver._fit_data(reg_basis, extended)
        assert residuals.shape == (n_pixels,)

    def test_coefficients_are_all_non_negative(
        self, nnls_solver, b_values, synthetic_multi
    ):
        """_fit_data returns only non-negative coefficients for all pixels."""
        signals, param_sets = synthetic_multi
        n_pixels = len(param_sets)
        nnls_solver.n_pixels = n_pixels
        reg_basis = nnls_solver._build_regularized_basis(b_values)
        extended = nnls_solver._extend_signal(signals)
        coeffs, _ = nnls_solver._fit_data(reg_basis, extended)
        assert np.all(coeffs >= 0), "All _fit_data coefficients must be non-negative"


# ---------------------------------------------------------------------------
# TestNNLSSolverRegularizationEffect
# ---------------------------------------------------------------------------


class TestNNLSSolverRegularizationEffect:
    """Tests comparing the behavioral effect of different regularization orders."""

    @pytest.mark.parametrize(
        "reg_order", [0, 1, 2, 3], ids=["none", "order1", "order2", "order3"]
    )
    def test_fit_completes_for_all_reg_orders(
        self, d_range, n_bins, b_values, synthetic_single, reg_order
    ):
        """fit() completes successfully and populates params_ for all supported reg_orders."""
        solver = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins),
            reg_order=reg_order,
            mu=0.02,
        )
        signal, _, _ = synthetic_single
        solver.fit(b_values, signal)
        assert "coefficients" in solver.params_

    def test_regularization_produces_smoother_spectrum(
        self, d_range, n_bins, b_values, synthetic_single
    ):
        """Second-order regularization produces a smoother coefficient spectrum than no regularization."""
        signal, _, _ = synthetic_single
        solver_unreg = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=0, mu=0.0
        )
        solver_reg = NNLSSolver(
            model=NNLSModel(d_range=d_range, n_bins=n_bins), reg_order=2, mu=0.1
        )
        solver_unreg.fit(b_values, signal)
        solver_reg.fit(b_values, signal)
        coeffs_unreg = solver_unreg.params_["coefficients"][0]
        coeffs_reg = solver_reg.params_["coefficients"][0]
        # Total variation as proxy for smoothness
        tv_unreg = np.sum(np.abs(np.diff(coeffs_unreg)))
        tv_reg = np.sum(np.abs(np.diff(coeffs_reg)))
        assert tv_reg <= tv_unreg, (
            f"Regularized spectrum (TV={tv_reg:.4f}) should be smoother than "
            f"unregularized (TV={tv_unreg:.4f})"
        )


# ---------------------------------------------------------------------------
# TestNNLSSolverDiagnostics
# ---------------------------------------------------------------------------


class TestNNLSSolverDiagnostics:
    """Tests for diagnostics content and isolation of get_params / get_diagnostics."""

    def test_get_params_returns_independent_copy(
        self, nnls_solver, b_values, synthetic_multi
    ):
        """get_params() returns an independent copy — mutating it leaves params_ unchanged."""
        signals, _ = synthetic_multi
        nnls_solver.fit(b_values, signals)
        original_shape = nnls_solver.params_["coefficients"].shape
        params_copy = nnls_solver.get_params()
        params_copy["coefficients"] = np.zeros((1, 1))
        assert nnls_solver.params_["coefficients"].shape == original_shape, (
            "Mutating the copy must not change params_"
        )

    def test_get_diagnostics_returns_independent_copy(
        self, nnls_solver, b_values, synthetic_single
    ):
        """get_diagnostics() returns an independent copy — mutating it leaves diagnostics_ unchanged."""
        signal, _, _ = synthetic_single
        nnls_solver.fit(b_values, signal)
        original_residual = nnls_solver.diagnostics_["residual"].copy()
        diag_copy = nnls_solver.get_diagnostics()
        diag_copy["residual"] = np.array([99999.0])
        np.testing.assert_array_equal(
            nnls_solver.diagnostics_["residual"],
            original_residual,
            err_msg="Mutating the diagnostics copy must not change diagnostics_",
        )

    def test_fit_returns_self_for_chaining(
        self, nnls_solver, b_values, synthetic_single
    ):
        """fit() returns the solver itself to support method chaining."""
        signal, _, _ = synthetic_single
        returned = nnls_solver.fit(b_values, signal)
        assert returned is nnls_solver, "fit() must return self"
