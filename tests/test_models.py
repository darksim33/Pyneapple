"""Tests for the exponential diffusion model classes (MonoExpModel, BiExpModel, TriExpModel)."""

import numpy as np
import pytest

from pyneapple.models import BiExpModel, MonoExpModel, NNLSModel, TriExpModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=float)


@pytest.fixture
def b_values():
    """Standard b-value array used across model tests."""
    return B_VALUES.copy()


# ---------------------------------------------------------------------------
# MonoExpModel
# ---------------------------------------------------------------------------


class TestMonoExpModel:
    """Test suite for the MonoExponential diffusion model."""

    # --- Initialization ---

    def test_init_default(self):
        """MonoExpModel initializes with no arguments and correct defaults."""
        model = MonoExpModel()
        assert model.fit_t1 is False
        assert model.fit_t1_steam is False
        assert model.repetition_time is None
        assert model.mixing_time is None

    def test_init_with_t1(self):
        """MonoExpModel with fit_t1=True stores repetition_time."""
        model = MonoExpModel(fit_t1=True, repetition_time=3000.0)
        assert model.fit_t1 is True
        assert model.repetition_time == 3000.0

    def test_init_steam_implies_t1(self):
        """STEAM mode automatically enables standard T1 correction."""
        model = MonoExpModel(
            fit_t1_steam=True, repetition_time=3000.0, mixing_time=50.0
        )
        assert model.fit_t1 is True
        assert model.fit_t1_steam is True

    def test_init_t1_missing_tr_raises(self):
        """ValueError raised when fit_t1=True but repetition_time is omitted."""
        with pytest.raises(ValueError, match="repetition_time is required"):
            MonoExpModel(fit_t1=True)

    def test_init_steam_missing_tm_raises(self):
        """ValueError raised when fit_t1_steam=True but mixing_time is omitted."""
        with pytest.raises(ValueError, match="mixing_time is required"):
            MonoExpModel(fit_t1_steam=True, repetition_time=3000.0)

    # --- param_names / n_params ---

    def test_param_names_default(self):
        """Default param_names are ['S0', 'D']."""
        model = MonoExpModel()
        assert model.param_names == ["S0", "D"]

    def test_param_names_with_t1(self):
        """T1 mode appends 'T1' to param_names."""
        model = MonoExpModel(fit_t1=True, repetition_time=3000.0)
        assert model.param_names == ["S0", "D", "T1"]

    def test_n_params_default(self):
        """Default model has 2 parameters."""
        model = MonoExpModel()
        assert model.n_params == 2

    def test_n_params_with_t1(self):
        """T1 model has 3 parameters."""
        model = MonoExpModel(fit_t1=True, repetition_time=3000.0)
        assert model.n_params == 3

    # --- forward ---

    def test_forward_output_shape(self, b_values):
        """forward() returns an array with the same length as b_values."""
        model = MonoExpModel()
        signal = model.forward(b_values, 1000.0, 0.001)
        assert signal.shape == b_values.shape

    def test_forward_values(self, b_values):
        """forward() matches the analytical monoexponential formula."""
        model = MonoExpModel()
        S0, D = 1000.0, 0.001
        expected = S0 * np.exp(-b_values * D)
        signal = model.forward(b_values, S0, D)
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_b0_equals_s0(self, b_values):
        """Signal at b=0 equals S0 (no diffusion weighting)."""
        model = MonoExpModel()
        S0, D = 850.0, 0.002
        signal = model.forward(b_values, S0, D)
        assert signal[0] == pytest.approx(S0)

    def test_forward_signal_decays_with_b(self, b_values):
        """Signal is strictly decreasing as b-value increases."""
        model = MonoExpModel()
        signal = model.forward(b_values, 1000.0, 0.001)
        assert np.all(np.diff(signal) < 0), "Signal must decay monotonically with b"

    def test_forward_with_t1_correction(self, b_values):
        """T1 mode scales signal by the T1 saturation factor."""
        TR, T1 = 3000.0, 1000.0
        model = MonoExpModel(fit_t1=True, repetition_time=TR)
        S0, D = 1000.0, 0.001
        signal = model.forward(b_values, S0, D, T1)
        t1_factor = 1 - np.exp(-TR / T1)
        expected = S0 * np.exp(-b_values * D) * t1_factor
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_with_steam_correction(self, b_values):
        """STEAM mode applies both T1 saturation and T1 mixing-time decay."""
        TR, TM, T1 = 3000.0, 50.0, 1000.0
        model = MonoExpModel(fit_t1_steam=True, repetition_time=TR, mixing_time=TM)
        S0, D = 1000.0, 0.001
        signal = model.forward(b_values, S0, D, T1)
        t1_factor = (1 - np.exp(-TR / T1)) * np.exp(-TM / T1)
        expected = S0 * np.exp(-b_values * D) * t1_factor
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    # --- jacobian ---

    def test_jacobian_output_shape_default(self, b_values):
        """jacobian() returns shape (n_b, n_params) for the default model."""
        model = MonoExpModel()
        jac = model.jacobian(b_values, 1000.0, 0.001)
        assert jac is not None
        assert jac.shape == (len(b_values), model.n_params)

    def test_jacobian_output_shape_t1(self, b_values):
        """jacobian() returns shape (n_b, n_params) for the T1 model."""
        model = MonoExpModel(fit_t1=True, repetition_time=3000.0)
        jac = model.jacobian(b_values, 1000.0, 0.001, 1000.0)
        assert jac is not None
        assert jac.shape == (len(b_values), model.n_params)

    # --- residual ---

    def test_residual_shape(self, b_values):
        """residual() returns an array of the same shape as the measured signal."""
        model = MonoExpModel()
        S0, D = 1000.0, 0.001
        measured = model.forward(b_values, S0, D) + 5.0
        residuals = model.residual(b_values, measured, np.array([S0, D]))
        assert residuals.shape == b_values.shape

    def test_residual_zero_for_perfect_params(self, b_values):
        """residual() is zero when measured signal matches the forward model exactly."""
        model = MonoExpModel()
        S0, D = 1000.0, 0.001
        perfect_signal = model.forward(b_values, S0, D)
        residuals = model.residual(b_values, perfect_signal, np.array([S0, D]))
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# BiExpModel
# ---------------------------------------------------------------------------


class TestBiExpModel:
    """Test suite for the Biexponential (IVIM) diffusion model."""

    # --- Initialization ---

    def test_init_default(self):
        """BiExpModel initializes in reduced mode by default."""
        model = BiExpModel()
        assert model.fit_reduced is True
        assert model.fit_s0 is False
        assert model.fit_t1 is False

    def test_init_full_mode(self):
        """BiExpModel can be initialized in full (non-reduced) mode."""
        model = BiExpModel(fit_reduced=False)
        assert model.fit_reduced is False

    def test_init_s0_mode(self):
        """BiExpModel S0 mode requires fit_reduced=True."""
        model = BiExpModel(fit_s0=True)
        assert model.fit_s0 is True
        assert model.fit_reduced is True

    def test_init_s0_with_full_raises(self):
        """ValueError raised when fit_s0=True and fit_reduced=False."""
        with pytest.raises(ValueError, match="fit_s0=True requires fit_reduced=True"):
            BiExpModel(fit_s0=True, fit_reduced=False)

    def test_init_t1_missing_tr_raises(self):
        """ValueError raised when fit_t1=True but repetition_time is omitted."""
        with pytest.raises(ValueError, match="repetition_time is required"):
            BiExpModel(fit_t1=True)

    def test_init_steam_missing_tm_raises(self):
        """ValueError raised when fit_t1_steam=True but mixing_time is omitted."""
        with pytest.raises(ValueError, match="mixing_time is required"):
            BiExpModel(fit_t1_steam=True, repetition_time=3000.0)

    # --- param_names / n_params ---

    @pytest.mark.parametrize(
        "kwargs, expected_names",
        [
            ({}, ["f1", "D1", "D2"]),
            ({"fit_reduced": False}, ["f1", "D1", "f2", "D2"]),
            ({"fit_s0": True}, ["f1", "D1", "D2", "S0"]),
            ({"fit_t1": True, "repetition_time": 3000.0}, ["f1", "D1", "D2", "T1"]),
        ],
        ids=["reduced", "full", "s0", "reduced+t1"],
    )
    def test_param_names(self, kwargs, expected_names):
        """param_names matches the expected list for each operating mode."""
        model = BiExpModel(**kwargs)
        assert model.param_names == expected_names

    @pytest.mark.parametrize(
        "kwargs, expected_n",
        [
            ({}, 3),
            ({"fit_reduced": False}, 4),
            ({"fit_s0": True}, 4),
            ({"fit_t1": True, "repetition_time": 3000.0}, 4),
        ],
        ids=["reduced", "full", "s0", "reduced+t1"],
    )
    def test_n_params(self, kwargs, expected_n):
        """n_params is consistent with the number of param_names entries."""
        model = BiExpModel(**kwargs)
        assert model.n_params == expected_n

    # --- forward ---

    def test_forward_output_shape_reduced(self, b_values):
        """forward() returns array of same length as b_values (reduced mode)."""
        model = BiExpModel()
        signal = model.forward(b_values, 0.3, 0.01, 0.001)
        assert signal.shape == b_values.shape

    def test_forward_values_reduced(self, b_values):
        """Reduced forward() matches biexponential formula with f2 = 1 - f1."""
        model = BiExpModel()
        f1, D1, D2 = 0.3, 0.01, 0.001
        expected = f1 * np.exp(-b_values * D1) + (1 - f1) * np.exp(-b_values * D2)
        signal = model.forward(b_values, f1, D1, D2)
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_values_full(self, b_values):
        """Full forward() matches biexponential formula with independent fractions."""
        model = BiExpModel(fit_reduced=False)
        f1, D1, f2, D2 = 0.3, 0.01, 0.5, 0.001
        expected = f1 * np.exp(-b_values * D1) + f2 * np.exp(-b_values * D2)
        signal = model.forward(b_values, f1, D1, f2, D2)
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_values_s0(self, b_values):
        """S0 forward() scales reduced signal by S0 amplitude."""
        model = BiExpModel(fit_s0=True)
        f1, D1, D2, S0 = 0.3, 0.01, 0.001, 1000.0
        expected = S0 * (
            f1 * np.exp(-b_values * D1) + (1 - f1) * np.exp(-b_values * D2)
        )
        signal = model.forward(b_values, f1, D1, D2, S0)
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_b0_reduced(self, b_values):
        """Reduced signal at b=0 equals 1.0 (fractions sum to 1)."""
        model = BiExpModel()
        signal = model.forward(b_values, 0.3, 0.01, 0.001)
        assert signal[0] == pytest.approx(1.0)

    def test_forward_with_t1_correction(self, b_values):
        """T1 correction scales biexponential signal by saturation factor."""
        TR, T1 = 3000.0, 1000.0
        model = BiExpModel(fit_t1=True, repetition_time=TR)
        f1, D1, D2 = 0.3, 0.01, 0.001
        signal = model.forward(b_values, f1, D1, D2, T1)
        t1_factor = 1 - np.exp(-TR / T1)
        base = f1 * np.exp(-b_values * D1) + (1 - f1) * np.exp(-b_values * D2)
        np.testing.assert_allclose(signal, base * t1_factor, rtol=1e-10)

    # --- jacobian ---

    @pytest.mark.parametrize(
        "kwargs, params",
        [
            ({}, (0.3, 0.01, 0.001)),
            ({"fit_reduced": False}, (0.3, 0.01, 0.5, 0.001)),
            ({"fit_s0": True}, (0.3, 0.01, 0.001, 1000.0)),
            ({"fit_t1": True, "repetition_time": 3000.0}, (0.3, 0.01, 0.001, 1000.0)),
        ],
        ids=["reduced", "full", "s0", "reduced+t1"],
    )
    def test_jacobian_output_shape(self, b_values, kwargs, params):
        """jacobian() returns shape (n_b, n_params) for each operating mode."""
        model = BiExpModel(**kwargs)
        jac = model.jacobian(b_values, *params)
        assert jac is not None
        assert jac.shape == (len(b_values), model.n_params)

    # --- residual ---

    def test_residual_zero_for_perfect_params(self, b_values):
        """residual() is zero when measured signal matches the forward model exactly."""
        model = BiExpModel()
        params = np.array([0.3, 0.01, 0.001])
        perfect_signal = model.forward(b_values, *params)
        residuals = model.residual(b_values, perfect_signal, params)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TriExpModel
# ---------------------------------------------------------------------------


class TestTriExpModel:
    """Test suite for the Triexponential diffusion model."""

    # --- Initialization ---

    def test_init_default(self):
        """TriExpModel initializes in reduced mode by default."""
        model = TriExpModel()
        assert model.fit_reduced is True
        assert model.fit_s0 is False
        assert model.fit_t1 is False

    def test_init_full_mode(self):
        """TriExpModel can be initialized in full (non-reduced) mode."""
        model = TriExpModel(fit_reduced=False)
        assert model.fit_reduced is False

    def test_init_s0_mode(self):
        """TriExpModel S0 mode requires fit_reduced=True."""
        model = TriExpModel(fit_s0=True)
        assert model.fit_s0 is True
        assert model.fit_reduced is True

    def test_init_s0_with_full_raises(self):
        """ValueError raised when fit_s0=True and fit_reduced=False."""
        with pytest.raises(ValueError, match="fit_s0=True requires fit_reduced=True"):
            TriExpModel(fit_s0=True, fit_reduced=False)

    def test_init_t1_missing_tr_raises(self):
        """ValueError raised when fit_t1=True but repetition_time is omitted."""
        with pytest.raises(ValueError, match="repetition_time is required"):
            TriExpModel(fit_t1=True)

    def test_init_steam_missing_tm_raises(self):
        """ValueError raised when fit_t1_steam=True but mixing_time is omitted."""
        with pytest.raises(ValueError, match="mixing_time is required"):
            TriExpModel(fit_t1_steam=True, repetition_time=3000.0)

    # --- param_names / n_params ---

    @pytest.mark.parametrize(
        "kwargs, expected_names",
        [
            ({}, ["f1", "D1", "f2", "D2", "D3"]),
            ({"fit_reduced": False}, ["f1", "D1", "f2", "D2", "f3", "D3"]),
            ({"fit_s0": True}, ["f1", "D1", "f2", "D2", "D3", "S0"]),
            (
                {"fit_t1": True, "repetition_time": 3000.0},
                ["f1", "D1", "f2", "D2", "D3", "T1"],
            ),
        ],
        ids=["reduced", "full", "s0", "reduced+t1"],
    )
    def test_param_names(self, kwargs, expected_names):
        """param_names matches the expected list for each operating mode."""
        model = TriExpModel(**kwargs)
        assert model.param_names == expected_names

    @pytest.mark.parametrize(
        "kwargs, expected_n",
        [
            ({}, 5),
            ({"fit_reduced": False}, 6),
            ({"fit_s0": True}, 6),
            ({"fit_t1": True, "repetition_time": 3000.0}, 6),
        ],
        ids=["reduced", "full", "s0", "reduced+t1"],
    )
    def test_n_params(self, kwargs, expected_n):
        """n_params matches the count of param_names entries for each mode."""
        model = TriExpModel(**kwargs)
        assert model.n_params == expected_n

    # --- forward ---

    def test_forward_output_shape_reduced(self, b_values):
        """forward() returns array of same length as b_values (reduced mode)."""
        model = TriExpModel()
        signal = model.forward(b_values, 0.2, 0.01, 0.3, 0.003, 0.001)
        assert signal.shape == b_values.shape

    def test_forward_values_reduced(self, b_values):
        """Reduced forward() matches triexponential formula with f3 = 1 - f1 - f2."""
        model = TriExpModel()
        f1, D1, f2, D2, D3 = 0.2, 0.01, 0.3, 0.003, 0.001
        f3 = 1 - f1 - f2
        expected = (
            f1 * np.exp(-b_values * D1)
            + f2 * np.exp(-b_values * D2)
            + f3 * np.exp(-b_values * D3)
        )
        signal = model.forward(b_values, f1, D1, f2, D2, D3)
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_values_full(self, b_values):
        """Full forward() matches triexponential formula with independent fractions."""
        model = TriExpModel(fit_reduced=False)
        f1, D1, f2, D2, f3, D3 = 0.2, 0.01, 0.3, 0.003, 0.4, 0.001
        expected = (
            f1 * np.exp(-b_values * D1)
            + f2 * np.exp(-b_values * D2)
            + f3 * np.exp(-b_values * D3)
        )
        signal = model.forward(b_values, f1, D1, f2, D2, f3, D3)
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_values_s0(self, b_values):
        """S0 forward() scales reduced triexponential signal by S0 amplitude."""
        model = TriExpModel(fit_s0=True)
        f1, D1, f2, D2, D3, S0 = 0.2, 0.01, 0.3, 0.003, 0.001, 1000.0
        f3 = 1 - f1 - f2
        expected = S0 * (
            f1 * np.exp(-b_values * D1)
            + f2 * np.exp(-b_values * D2)
            + f3 * np.exp(-b_values * D3)
        )
        signal = model.forward(b_values, f1, D1, f2, D2, D3, S0)
        np.testing.assert_allclose(signal, expected, rtol=1e-10)

    def test_forward_b0_reduced(self, b_values):
        """Reduced signal at b=0 equals 1.0 (fractions sum to 1)."""
        model = TriExpModel()
        signal = model.forward(b_values, 0.2, 0.01, 0.3, 0.003, 0.001)
        assert signal[0] == pytest.approx(1.0)

    def test_forward_with_t1_correction(self, b_values):
        """T1 correction scales triexponential signal by saturation factor."""
        TR, T1 = 3000.0, 1000.0
        model = TriExpModel(fit_t1=True, repetition_time=TR)
        f1, D1, f2, D2, D3 = 0.2, 0.01, 0.3, 0.003, 0.001
        signal = model.forward(b_values, f1, D1, f2, D2, D3, T1)
        t1_factor = 1 - np.exp(-TR / T1)
        f3 = 1 - f1 - f2
        base = (
            f1 * np.exp(-b_values * D1)
            + f2 * np.exp(-b_values * D2)
            + f3 * np.exp(-b_values * D3)
        )
        np.testing.assert_allclose(signal, base * t1_factor, rtol=1e-10)

    # --- jacobian ---

    @pytest.mark.parametrize(
        "kwargs, params",
        [
            ({}, (0.2, 0.01, 0.3, 0.003, 0.001)),
            ({"fit_reduced": False}, (0.2, 0.01, 0.3, 0.003, 0.4, 0.001)),
            ({"fit_s0": True}, (0.2, 0.01, 0.3, 0.003, 0.001, 1000.0)),
            (
                {"fit_t1": True, "repetition_time": 3000.0},
                (0.2, 0.01, 0.3, 0.003, 0.001, 1000.0),
            ),
        ],
        ids=["reduced", "full", "s0", "reduced+t1"],
    )
    def test_jacobian_output_shape(self, b_values, kwargs, params):
        """jacobian() returns shape (n_b, n_params) for each operating mode."""
        model = TriExpModel(**kwargs)
        jac = model.jacobian(b_values, *params)
        assert jac is not None
        assert jac.shape == (len(b_values), model.n_params)

    # --- residual ---

    def test_residual_zero_for_perfect_params(self, b_values):
        """residual() is zero when measured signal matches the forward model exactly."""
        model = TriExpModel()
        params = np.array([0.2, 0.01, 0.3, 0.003, 0.001])
        perfect_signal = model.forward(b_values, *params)
        residuals = model.residual(b_values, perfect_signal, params)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# NNLSModel
# ---------------------------------------------------------------------------

D_RANGE = (1e-4, 0.1)
N_BINS = 50


class TestNNLSModel:
    """Test suite for the NNLSModel distribution model."""

    # --- Initialization ---

    def test_init_stores_d_range(self):
        """NNLSModel stores the d_range argument."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        assert model.d_range == D_RANGE

    def test_init_stores_n_bins(self):
        """NNLSModel stores the n_bins argument."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        assert model.n_bins == N_BINS

    # --- bins ---

    def test_bins_shape(self):
        """bins property returns a 1-D array of length n_bins."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        assert model.bins.shape == (N_BINS,)

    def test_bins_start_at_d_min(self):
        """First bin value equals d_min."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        assert model.bins[0] == pytest.approx(D_RANGE[0])

    def test_bins_end_at_d_max(self):
        """Last bin value equals d_max."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        assert model.bins[-1] == pytest.approx(D_RANGE[1])

    def test_bins_are_log_spaced(self):
        """Consecutive bin ratios are constant (logarithmic spacing)."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        log_bins = np.log10(model.bins)
        diffs = np.diff(log_bins)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)

    # --- get_basis ---

    def test_get_basis_shape(self, b_values):
        """get_basis returns matrix of shape (n_measurements, n_bins)."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        basis = model.get_basis(b_values)
        assert basis.shape == (len(b_values), N_BINS)

    def test_get_basis_at_b0_is_all_ones(self):
        """At b=0, all basis entries equal 1.0."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        basis = model.get_basis(np.array([0.0]))
        np.testing.assert_allclose(basis, np.ones((1, N_BINS)))

    def test_get_basis_raises_for_2d_input(self, b_values):
        """get_basis raises ValueError when xdata is not 1-D."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        with pytest.raises(ValueError):
            model.get_basis(b_values.reshape(2, -1))

    # --- forward ---

    def test_forward_reconstructs_signal(self, b_values):
        """forward() with spectrum coefficients reproduces basis @ spectrum."""
        model = NNLSModel(d_range=D_RANGE, n_bins=N_BINS)
        rng = np.random.default_rng(0)
        spectrum = rng.random(N_BINS)
        expected = model.get_basis(b_values) @ spectrum
        result = model.forward(b_values, *spectrum)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Fixed Parameters (ParametricModel base-class features)
# ---------------------------------------------------------------------------


class TestFixedParamsMonoExp:
    """Test fixed-param support exposed by ParametricModel, using MonoExpModel."""

    # --- param_names filtering ---

    def test_param_names_excludes_fixed(self):
        """param_names omits parameters listed in fixed_params."""
        model = MonoExpModel(fixed_params={"S0": 500.0})
        assert model.param_names == ["D"]

    def test_param_names_excludes_fixed_t1(self):
        """Fixing T1 on a T1-enabled model leaves ['S0', 'D']."""
        model = MonoExpModel(
            fit_t1=True, repetition_time=3000.0, fixed_params={"T1": 1200.0}
        )
        assert model.param_names == ["S0", "D"]

    def test_n_params_reflects_fixed(self):
        """n_params decreases when a parameter is fixed."""
        model = MonoExpModel(fixed_params={"D": 0.001})
        assert model.n_params == 1

    # --- validation ---

    def test_invalid_fixed_param_raises(self):
        """ValueError when a fixed param name is not in _all_param_names."""
        with pytest.raises(ValueError, match="Unknown fixed parameter"):
            MonoExpModel(fixed_params={"BOGUS": 42.0})

    def test_fix_all_params_raises(self):
        """ValueError when fixing every parameter (nothing left to fit)."""
        with pytest.raises(ValueError, match="Cannot fix all parameters"):
            MonoExpModel(fixed_params={"S0": 1.0, "D": 0.001})

    # --- forward_with_fixed ---

    def test_forward_with_fixed_injects_s0(self, b_values):
        """forward_with_fixed with fixed S0 produces the correct signal."""
        model = MonoExpModel()
        S0, D = 800.0, 0.001
        signal = model.forward_with_fixed(b_values, {"S0": S0}, D)
        expected = model.forward(b_values, S0, D)
        np.testing.assert_allclose(signal, expected, rtol=1e-12)

    def test_forward_with_fixed_injects_d(self, b_values):
        """forward_with_fixed with fixed D produces the correct signal."""
        model = MonoExpModel()
        S0, D = 1000.0, 0.002
        signal = model.forward_with_fixed(b_values, {"D": D}, S0)
        expected = model.forward(b_values, S0, D)
        np.testing.assert_allclose(signal, expected, rtol=1e-12)

    def test_forward_with_fixed_t1(self, b_values):
        """forward_with_fixed with fixed T1 on a T1-enabled model."""
        TR, T1 = 3000.0, 1000.0
        model = MonoExpModel(fit_t1=True, repetition_time=TR)
        S0, D = 900.0, 0.0015
        signal = model.forward_with_fixed(b_values, {"T1": T1}, S0, D)
        expected = model.forward(b_values, S0, D, T1)
        np.testing.assert_allclose(signal, expected, rtol=1e-12)

    # --- jacobian_with_fixed ---

    def test_jacobian_with_fixed_shape(self, b_values):
        """jacobian_with_fixed returns (n_b, n_free) columns."""
        model = MonoExpModel()
        jac = model.jacobian_with_fixed(b_values, {"S0": 1000.0}, 0.001)
        if jac is None:
            assert False, "Jacobian should not be None"
        assert jac.shape == (len(b_values), 1)  # only D is free

    def test_jacobian_with_fixed_values(self, b_values):
        """jacobian_with_fixed column matches the D column of the full Jacobian."""
        model = MonoExpModel()
        S0, D = 1000.0, 0.001
        jac_full = model.jacobian(b_values, S0, D)
        jac_fixed = model.jacobian_with_fixed(b_values, {"S0": S0}, D)
        # D is column 1 in the full Jacobian
        if jac_fixed is None or jac_full is None:
            assert False, "Jacobian should not be None"
        np.testing.assert_allclose(jac_fixed[:, 0], jac_full[:, 1], rtol=1e-12)

    def test_jacobian_with_fixed_t1_shape(self, b_values):
        """Fixing T1 on a 3-param model yields (n_b, 2) Jacobian."""
        model = MonoExpModel(fit_t1=True, repetition_time=3000.0)
        jac = model.jacobian_with_fixed(b_values, {"T1": 1000.0}, 900.0, 0.001)
        if jac is None:
            assert False, "Jacobian should not be None"
        assert jac.shape == (len(b_values), 2)  # S0 and D free


# ---------------------------------------------------------------------------
# ParametricModel.precondition, validate_params, validate_bounds
# ---------------------------------------------------------------------------


class TestParametricModelHelpers:
    """Coverage for precondition(), validate_params(), validate_bounds() on base class."""

    @pytest.mark.unit
    def test_precondition_none_returns_same(self, b_values):
        """precondition(method='none') returns the Jacobian unchanged."""
        model = MonoExpModel()
        jac = model.jacobian(b_values, 1000.0, 0.001)
        if jac is None:
            assert False, "Jacobian should not be None"
        result = model.precondition(jac, method="none")
        np.testing.assert_array_equal(result, jac)

    @pytest.mark.unit
    def test_precondition_diagonal(self, b_values):
        """precondition(method='diagonal') scale-normalises columns."""
        model = MonoExpModel()
        jac = model.jacobian(b_values, 1000.0, 0.001)
        if jac is None:
            assert False, "Jacobian should not be None"

        result = model.precondition(jac, method="diagonal")
        # Each column should have unit Euclidean norm (or norm ≤ 1 if all zeros)
        if result is None:
            assert False, "Preconditioned Jacobian should not be None"
        col_norms = np.sqrt(np.sum(result**2, axis=0))
        np.testing.assert_allclose(col_norms, 1.0, atol=1e-10)

    @pytest.mark.unit
    def test_precondition_unknown_raises(self, b_values):
        """precondition() raises ValueError for an unknown method name."""
        model = MonoExpModel()
        jac = model.jacobian(b_values, 1000.0, 0.001)
        if jac is None:
            assert False, "Jacobian should not be None"
        with pytest.raises(ValueError, match="Unknown preconditioning method"):
            model.precondition(jac, method="foo")

    @pytest.mark.unit
    def test_validate_params_valid(self):
        """validate_params returns a numpy array for a valid parameter dict."""
        model = MonoExpModel()
        result = model.validate_params({"S0": 1000.0, "D": 0.001})
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    @pytest.mark.unit
    def test_validate_params_missing_raises(self):
        """validate_params raises ValueError when a required parameter is absent."""
        model = MonoExpModel()
        with pytest.raises(ValueError, match="Missing required parameters"):
            model.validate_params({"S0": 1000.0})  # D missing

    @pytest.mark.unit
    def test_validate_params_extra_warns(self, recwarn):
        """validate_params does not raise but logs a warning for extra keys."""
        model = MonoExpModel()
        # Should not raise
        result = model.validate_params({"S0": 1000.0, "D": 0.001, "EXTRA": 9.9})
        assert result.shape == (2,)

    @pytest.mark.unit
    def test_validate_bounds_valid(self):
        """validate_bounds returns (lower, upper) arrays for a valid bounds dict."""
        model = MonoExpModel()
        lower, upper = model.validate_bounds({"S0": (0.0, 5000.0), "D": (0.0001, 0.1)})
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert lower[0] == pytest.approx(0.0)
        assert upper[1] == pytest.approx(0.1)

    @pytest.mark.unit
    def test_validate_bounds_missing_raises(self):
        """validate_bounds raises ValueError when a required bound is absent."""
        model = MonoExpModel()
        with pytest.raises(ValueError, match="Missing bounds"):
            model.validate_bounds({"S0": (0.0, 5000.0)})  # D missing

    @pytest.mark.unit
    def test_validate_bounds_extra_ignored(self):
        """validate_bounds does not raise for extra keys."""
        model = MonoExpModel()
        lower, upper = model.validate_bounds(
            {"S0": (0.0, 5000.0), "D": (0.0001, 0.1), "BOGUS": (0.0, 1.0)}
        )
        assert lower.shape == (2,)


class TestFixedParamsBiExp:
    """Spot-check fixed params work with a multi-mode model."""

    def test_fixed_d2_in_reduced_mode(self):
        """Fixing D2 in reduced BiExp leaves ['f1', 'D1']."""
        model = BiExpModel(fixed_params={"D2": 0.001})
        assert model.param_names == ["f1", "D1"]

    def test_forward_with_fixed_d2(self, b_values):
        """forward_with_fixed with fixed D2 matches manual forward call."""
        model = BiExpModel()
        f1, D1, D2 = 0.3, 0.01, 0.001
        signal = model.forward_with_fixed(b_values, {"D2": D2}, f1, D1)
        expected = model.forward(b_values, f1, D1, D2)
        np.testing.assert_allclose(signal, expected, rtol=1e-12)
