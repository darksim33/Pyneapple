"""Tests for CurveFitSolver — initialization, validation helpers, and fitting."""

import numpy as np
import pytest

from pyneapple.models import MonoExpModel, BiExpModel
from pyneapple.solvers import CurveFitSolver


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=float)


@pytest.fixture
def b_values():
    """Standard b-value array."""
    return B_VALUES.copy()


@pytest.fixture
def monoexp_model():
    """MonoExponential model (default, no T1)."""
    return MonoExpModel()


@pytest.fixture
def monoexp_p0():
    """Reasonable scalar initial guesses for MonoExpModel."""
    return {"S0": 900.0, "D": 0.0012}


@pytest.fixture
def monoexp_bounds():
    """Parameter bounds for MonoExpModel."""
    return {"S0": (0.0, 5000.0), "D": (0.0, 0.05)}


@pytest.fixture
def monoexp_solver(monoexp_model, monoexp_p0, monoexp_bounds):
    """Fully configured CurveFitSolver using the TRF method (supports bounds)."""
    return CurveFitSolver(
        model=monoexp_model,
        max_iter=2000,
        tol=1e-10,
        p0=monoexp_p0,
        bounds=monoexp_bounds,
        method="trf",
    )


@pytest.fixture
def synthetic_single(b_values):
    """Noise-free monoexponential signal for a single voxel."""
    S0_true, D_true = 1000.0, 0.001
    signal = MonoExpModel().forward(b_values, S0_true, D_true)
    return signal, S0_true, D_true


@pytest.fixture
def synthetic_multi(b_values):
    """Noise-free signals for 3 voxels with different parameter values."""
    param_sets = [
        (1000.0, 0.001),
        (800.0, 0.002),
        (1200.0, 0.0005),
    ]
    model = MonoExpModel()
    signals = np.array(
        [model.forward(b_values, S0, D) for S0, D in param_sets]
    )  # shape (3, 8)
    return signals, param_sets


# ---------------------------------------------------------------------------
# TestCurveFitSolverInit
# ---------------------------------------------------------------------------


class TestCurveFitSolverInit:
    """Test suite for CurveFitSolver construction and parameter validation."""

    def test_valid_initialization(self, monoexp_model, monoexp_p0, monoexp_bounds):
        """Solver initializes correctly with valid model, p0, and bounds."""
        solver = CurveFitSolver(
            model=monoexp_model,
            max_iter=500,
            tol=1e-8,
            p0=monoexp_p0,
            bounds=monoexp_bounds,
            method="trf",
        )
        assert solver.method == "trf"
        assert solver.max_iter == 500
        assert solver.tol == pytest.approx(1e-8)
        assert solver.p0 == monoexp_p0
        assert solver.bounds == monoexp_bounds

    def test_p0_none_raises_not_implemented(self, monoexp_model, monoexp_bounds):
        """NotImplementedError raised when p0 is None (defaults not yet implemented)."""
        with pytest.raises(NotImplementedError):
            CurveFitSolver(
                model=monoexp_model,
                max_iter=500,
                tol=1e-8,
                p0=None,
                bounds=monoexp_bounds,
            )

    def test_bounds_none_raises_not_implemented(self, monoexp_model, monoexp_p0):
        """NotImplementedError raised when bounds is None (defaults not yet implemented)."""
        with pytest.raises(NotImplementedError):
            CurveFitSolver(
                model=monoexp_model,
                max_iter=500,
                tol=1e-8,
                p0=monoexp_p0,
                bounds=None,
            )

    def test_p0_wrong_value_type_raises(self, monoexp_model, monoexp_bounds):
        """ValueError raised when p0 values are not scalars."""
        bad_p0 = {"S0": [900.0], "D": [0.0012]}  # lists, not scalars
        with pytest.raises(ValueError):
            CurveFitSolver(
                model=monoexp_model,
                max_iter=500,
                tol=1e-8,
                p0=bad_p0,
                bounds=monoexp_bounds,
            )

    def test_bounds_wrong_value_type_raises(self, monoexp_model, monoexp_p0):
        """ValueError raised when bounds values are not (lower, upper) tuples."""
        bad_bounds = {"S0": 0.0, "D": 0.0}  # scalars, not tuples
        with pytest.raises((ValueError, TypeError)):
            CurveFitSolver(
                model=monoexp_model,
                max_iter=500,
                tol=1e-8,
                p0=monoexp_p0,
                bounds=bad_bounds,
            )

    def test_p0_missing_parameter_raises(self, monoexp_model, monoexp_bounds):
        """ValueError raised when p0 is missing a required model parameter."""
        incomplete_p0 = {"S0": 900.0}  # missing 'D'
        with pytest.raises(ValueError, match="Missing"):
            CurveFitSolver(
                model=monoexp_model,
                max_iter=500,
                tol=1e-8,
                p0=incomplete_p0,
                bounds=monoexp_bounds,
            )

    def test_default_multithreading_disabled(
        self, monoexp_model, monoexp_p0, monoexp_bounds
    ):
        """Multi-threading is disabled by default."""
        solver = CurveFitSolver(
            model=monoexp_model,
            max_iter=500,
            tol=1e-8,
            p0=monoexp_p0,
            bounds=monoexp_bounds,
        )
        assert solver.multi_threading is False

    def test_params_and_diagnostics_none_before_fit(
        self, monoexp_model, monoexp_p0, monoexp_bounds
    ):
        """params_ and diagnostics_ are None before fit() is called."""
        solver = CurveFitSolver(
            model=monoexp_model,
            max_iter=500,
            tol=1e-8,
            p0=monoexp_p0,
            bounds=monoexp_bounds,
        )
        assert len(solver.params_) == 0
        assert len(solver.diagnostics_) == 0


# ---------------------------------------------------------------------------
# TestCurveFitSolverBeforeFit
# ---------------------------------------------------------------------------


class TestCurveFitSolverBeforeFit:
    """Tests for BaseSolver helpers before fit() is called."""

    def test_get_params_raises_before_fit(self, monoexp_solver):
        """get_params() raises RuntimeError when called before fit()."""
        with pytest.raises(RuntimeError, match="No parameters available"):
            monoexp_solver.get_params()

    def test_get_diagnostics_raises_before_fit(self, monoexp_solver):
        """get_diagnostics() raises RuntimeError when called before fit()."""
        with pytest.raises(RuntimeError, match="No diagnostics available"):
            monoexp_solver.get_diagnostics()


# ---------------------------------------------------------------------------
# TestValidateP0AndBounds
# ---------------------------------------------------------------------------


class TestValidateP0AndBounds:
    """Tests for CurveFitSolver._validate_p0_and_bounds()."""

    def test_dict_p0_transforms_to_correct_shape(
        self, monoexp_solver, monoexp_p0, monoexp_bounds
    ):
        """Dict p0 is transformed to ndarray of shape (n_params, n_pixels)."""
        n_pixels = 5
        p0_arr, _ = monoexp_solver._validate_p0_and_bounds(
            monoexp_p0, monoexp_bounds, n_pixels
        )
        assert isinstance(p0_arr, np.ndarray)
        assert p0_arr.shape == (
            2,
            n_pixels,
        ), f"Expected (2, {n_pixels}), got {p0_arr.shape}"

    def test_dict_bounds_transform_to_correct_shape(
        self, monoexp_solver, monoexp_p0, monoexp_bounds
    ):
        """Dict bounds are transformed to two ndarrays of shape (n_params, n_pixels)."""
        n_pixels = 5
        _, bounds_arr = monoexp_solver._validate_p0_and_bounds(
            monoexp_p0, monoexp_bounds, n_pixels
        )
        assert bounds_arr[0].shape == (2, n_pixels)
        assert bounds_arr[1].shape == (2, n_pixels)

    def test_none_p0_uses_solver_defaults(self, monoexp_solver, monoexp_bounds):
        """When p0=None, solver default p0 is used."""
        n_pixels = 3
        p0_arr, _ = monoexp_solver._validate_p0_and_bounds(
            None, monoexp_bounds, n_pixels
        )
        assert p0_arr.shape == (2, n_pixels)

    def test_none_bounds_uses_solver_defaults(self, monoexp_solver, monoexp_p0):
        """When bounds=None, solver default bounds are used."""
        n_pixels = 3
        _, bounds_arr = monoexp_solver._validate_p0_and_bounds(
            monoexp_p0, None, n_pixels
        )
        assert bounds_arr[0].shape == (2, n_pixels)
        assert bounds_arr[1].shape == (2, n_pixels)

    def test_ndarray_p0_passed_through(self, monoexp_solver, monoexp_bounds):
        """ndarray p0 of correct shape passes through unchanged."""
        n_pixels = 4
        p0_arr = np.tile(np.array([900.0, 0.0012])[:, None], (1, n_pixels))
        result_p0, _ = monoexp_solver._validate_p0_and_bounds(
            p0_arr, monoexp_bounds, n_pixels
        )
        np.testing.assert_array_equal(result_p0, p0_arr)

    def test_p0_shape_mismatch_raises(self, monoexp_solver, monoexp_bounds):
        """ValueError raised when p0 ndarray column count doesn't match n_pixels."""
        p0_wrong = np.zeros((2, 3))  # 3 pixels but n_pixels=5
        with pytest.raises(ValueError, match="shape"):
            monoexp_solver._validate_p0_and_bounds(p0_wrong, monoexp_bounds, n_pixels=5)

    def test_bounds_shape_mismatch_raises(self, monoexp_solver, monoexp_p0):
        """ValueError raised when bounds ndarray columns don't match n_pixels."""
        bad_bounds = (np.zeros((2, 3)), np.ones((2, 3)))  # 3 pixels but n_pixels=5
        with pytest.raises(ValueError, match="shape"):
            monoexp_solver._validate_p0_and_bounds(monoexp_p0, bad_bounds, n_pixels=5)

    def test_p0_not_dict_or_ndarray_raises(self, monoexp_solver, monoexp_bounds):
        """ValueError raised when p0 is an unsupported type."""
        with pytest.raises(ValueError):
            monoexp_solver._validate_p0_and_bounds(
                [900.0, 0.0012], monoexp_bounds, n_pixels=1
            )


# ---------------------------------------------------------------------------
# TestCurveFitSolverFitSinglePixel
# ---------------------------------------------------------------------------


class TestCurveFitSolverFitSinglePixel:
    """Tests for CurveFitSolver._fit_single_pixel()."""

    def test_returns_tuple_of_two_arrays(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """_fit_single_pixel returns (popt, pcov) both as ndarrays."""
        signal, _, _ = synthetic_single
        p0 = np.array([900.0, 0.0012])
        bounds = (np.array([0.0, 0.0]), np.array([5000.0, 0.05]))
        popt, pcov = monoexp_solver._fit_single_pixel(b_values, signal, p0, bounds)
        assert isinstance(popt, np.ndarray)
        assert isinstance(pcov, np.ndarray)

    def test_converges_to_true_params(self, monoexp_solver, b_values, synthetic_single):
        """_fit_single_pixel recovers the true parameters from noise-free data."""
        signal, S0_true, D_true = synthetic_single
        p0 = np.array([900.0, 0.0012])
        bounds = (np.array([0.0, 0.0]), np.array([5000.0, 0.05]))
        popt, _ = monoexp_solver._fit_single_pixel(b_values, signal, p0, bounds)
        np.testing.assert_allclose(popt[0], S0_true, rtol=1e-3)
        np.testing.assert_allclose(popt[1], D_true, rtol=1e-3)

    def test_failed_fit_returns_p0_and_nan_cov(
        self, monoexp_solver, b_values, synthetic_single, mocker
    ):
        """When curve_fit raises RuntimeError, _fit_single_pixel returns p0 and NaN covariance."""
        signal, _, _ = synthetic_single
        p0 = np.array([900.0, 0.0012])
        bounds = (np.array([0.0, 0.0]), np.array([5000.0, 0.05]))

        mocker.patch(
            "pyneapple.solvers.curvefit.curve_fit",
            side_effect=RuntimeError(
                "Optimal parameters not found: too many iterations"
            ),
        )

        popt, pcov = monoexp_solver._fit_single_pixel(b_values, signal, p0, bounds)
        np.testing.assert_array_equal(popt, p0)
        assert np.all(np.isnan(pcov)), "Failed fit should return NaN covariance"

    def test_popt_shape(self, monoexp_solver, b_values, synthetic_single):
        """popt has length equal to n_params."""
        signal, _, _ = synthetic_single
        p0 = np.array([900.0, 0.0012])
        bounds = (np.array([0.0, 0.0]), np.array([5000.0, 0.05]))
        popt, _ = monoexp_solver._fit_single_pixel(b_values, signal, p0, bounds)
        assert popt.shape == (monoexp_solver.model.n_params,)


# ---------------------------------------------------------------------------
# TestCurveFitSolverFit
# ---------------------------------------------------------------------------


class TestCurveFitSolverFit:
    """End-to-end tests for CurveFitSolver.fit()."""

    @pytest.mark.unit
    def test_fit_single_voxel_returns_self(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """fit() returns the solver instance itself to enable method chaining."""
        signal, _, _ = synthetic_single
        result = monoexp_solver.fit(b_values, signal)
        assert result is monoexp_solver

    @pytest.mark.unit
    def test_fit_single_voxel_recovers_params(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """fit() recovers true parameters from noise-free single-voxel data."""
        signal, S0_true, D_true = synthetic_single
        monoexp_solver.fit(b_values, signal)
        params = monoexp_solver.get_params()
        assert params["S0"] == pytest.approx(S0_true, rel=1e-3)
        assert params["D"] == pytest.approx(D_true, rel=1e-3)

    @pytest.mark.unit
    def test_fit_stores_params_and_diagnostics(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """After fit(), params_ and diagnostics_ are populated."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(b_values, signal)
        assert monoexp_solver.params_ is not None
        assert monoexp_solver.diagnostics_ is not None

    @pytest.mark.unit
    def test_get_params_after_fit(self, monoexp_solver, b_values, synthetic_single):
        """get_params() returns a copy of fitted parameters after fit()."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(b_values, signal)
        params = monoexp_solver.get_params()
        assert isinstance(params, dict)
        assert "S0" in params

    @pytest.mark.unit
    def test_get_diagnostics_after_fit(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """get_diagnostics() returns diagnostics dict after fit()."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(b_values, signal)
        diag = monoexp_solver.get_diagnostics()
        assert "pcov" in diag
        assert "n_pixels" in diag

    @pytest.mark.unit
    def test_fit_multi_voxel_params_are_arrays(
        self, monoexp_solver, b_values, synthetic_multi
    ):
        """After multi-voxel fit, params_ values are arrays of shape (n_voxels,)."""
        signals, param_sets = synthetic_multi
        monoexp_solver.fit(b_values, signals)
        params = monoexp_solver.get_params()
        assert isinstance(params["S0"], np.ndarray)
        assert params["S0"].shape == (3,)
        assert params["D"].shape == (3,)

    @pytest.mark.unit
    def test_fit_multi_voxel_recovers_params(
        self, monoexp_solver, b_values, synthetic_multi
    ):
        """fit() recovers true parameters for each voxel in multi-voxel data."""
        signals, param_sets = synthetic_multi
        monoexp_solver.fit(b_values, signals)
        params = monoexp_solver.get_params()
        for i, (S0_true, D_true) in enumerate(param_sets):
            assert params["S0"][i] == pytest.approx(S0_true, rel=1e-2), (
                f"Voxel {i}: S0 mismatch"
            )
            assert params["D"][i] == pytest.approx(D_true, rel=1e-2), (
                f"Voxel {i}: D mismatch"
            )

    @pytest.mark.unit
    def test_fit_resets_state_on_second_call(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """Calling fit() a second time overwrites previous results cleanly."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(b_values, signal)
        first_s0 = monoexp_solver.params_["S0"]
        monoexp_solver.fit(b_values, signal * 0.5)  # different data
        second_s0 = monoexp_solver.params_["S0"]
        assert first_s0 != pytest.approx(second_s0, rel=0.01), (
            "Params should differ after re-fitting different data"
        )

    @pytest.mark.unit
    def test_fit_shape_mismatch_raises(self, monoexp_solver, b_values):
        """ValueError raised when ydata length does not match xdata."""
        bad_ydata = np.ones(len(b_values) + 1)
        with pytest.raises(ValueError):
            monoexp_solver.fit(b_values, bad_ydata)


# ---------------------------------------------------------------------------
# Additional fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def biexp_model():
    """BiExponential model in default reduced mode."""
    return BiExpModel()


@pytest.fixture
def biexp_p0():
    """Reasonable scalar initial guesses for BiExpModel (reduced)."""
    return {"f1": 0.25, "D1": 0.008, "D2": 0.0008}


@pytest.fixture
def biexp_bounds():
    """Parameter bounds for BiExpModel (reduced)."""
    return {"f1": (0.0, 1.0), "D1": (0.0, 0.1), "D2": (0.0, 0.01)}


@pytest.fixture
def biexp_solver(biexp_model, biexp_p0, biexp_bounds):
    """Fully configured CurveFitSolver for BiExpModel."""
    return CurveFitSolver(
        model=biexp_model,
        max_iter=2000,
        tol=1e-10,
        p0=biexp_p0,
        bounds=biexp_bounds,
        method="trf",
    )


@pytest.fixture
def synthetic_biexp_single(b_values):
    """Noise-free biexponential signal for a single voxel."""
    f1_true, D1_true, D2_true = 0.3, 0.01, 0.001
    signal = BiExpModel().forward(b_values, f1_true, D1_true, D2_true)
    return signal, f1_true, D1_true, D2_true


# ---------------------------------------------------------------------------
# TestCurveFitSolverFitOverrides
# ---------------------------------------------------------------------------


class TestCurveFitSolverFitOverrides:
    """Tests for per-call p0 and bounds overrides passed directly to fit()."""

    def test_fit_with_dict_p0_override(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """fit() accepts a dict p0 override and uses it instead of solver defaults."""
        signal, S0_true, D_true = synthetic_single
        override_p0 = {"S0": 950.0, "D": 0.0011}
        monoexp_solver.fit(b_values, signal, p0=override_p0)
        params = monoexp_solver.get_params()
        assert params["S0"] == pytest.approx(S0_true, rel=1e-2)
        assert params["D"] == pytest.approx(D_true, rel=1e-2)

    def test_fit_with_ndarray_p0_override(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """fit() accepts an ndarray p0 override of shape (n_params, n_pixels)."""
        signal, S0_true, _ = synthetic_single
        override_p0 = np.array([[950.0], [0.0011]])  # (2, 1) for single voxel
        monoexp_solver.fit(b_values, signal, p0=override_p0)
        assert monoexp_solver.params_["S0"] == pytest.approx(S0_true, rel=1e-2)

    def test_fit_with_dict_bounds_override(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """fit() accepts a dict bounds override and uses it instead of solver defaults."""
        signal, S0_true, _ = synthetic_single
        tight_bounds = {"S0": (900.0, 1100.0), "D": (0.0005, 0.005)}
        monoexp_solver.fit(b_values, signal, bounds=tight_bounds)
        assert monoexp_solver.params_["S0"] == pytest.approx(S0_true, rel=1e-2)

    def test_fit_with_tuple_bounds_override(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """fit() accepts a pre-built (lower, upper) ndarray tuple as bounds override."""
        signal, _, _ = synthetic_single
        lower = np.array([[0.0], [0.0]])  # (n_params, 1)
        upper = np.array([[5000.0], [0.05]])
        monoexp_solver.fit(b_values, signal, bounds=(lower, upper))
        assert "S0" in monoexp_solver.get_params()

    def test_fit_per_call_override_does_not_mutate_solver_defaults(
        self, monoexp_solver, b_values, synthetic_single, monoexp_p0, monoexp_bounds
    ):
        """Passing overrides to fit() does not change solver.p0 or solver.bounds."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(
            b_values,
            signal,
            p0={"S0": 500.0, "D": 0.005},
            bounds={"S0": (0.0, 2000.0), "D": (0.0, 0.02)},
        )
        assert monoexp_solver.p0 == monoexp_p0
        assert monoexp_solver.bounds == monoexp_bounds

    def test_fit_invalid_p0_override_type_raises(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """ValueError raised when p0 override is an unsupported type (e.g. list)."""
        signal, _, _ = synthetic_single
        with pytest.raises(ValueError):
            monoexp_solver.fit(b_values, signal, p0=[900.0, 0.001])


# ---------------------------------------------------------------------------
# TestCurveFitSolverFitBiExp
# ---------------------------------------------------------------------------


class TestCurveFitSolverFitBiExp:
    """Tests for CurveFitSolver using a BiExponential model."""

    def test_biexp_param_names_stored(self, biexp_solver):
        """BiExpModel solver stores the correct parameter names."""
        assert set(biexp_solver.model.param_names) == {"f1", "D1", "D2"}
        assert biexp_solver.model.n_params == 3

    def test_biexp_fit_returns_all_params(
        self, biexp_solver, b_values, synthetic_biexp_single
    ):
        """fit() on BiExpModel populates params_ with all 3 parameter keys."""
        signal, *_ = synthetic_biexp_single
        biexp_solver.fit(b_values, signal)
        assert set(biexp_solver.get_params().keys()) == {"f1", "D1", "D2"}

    def test_biexp_fit_recovers_params(
        self, biexp_solver, b_values, synthetic_biexp_single
    ):
        """fit() recovers true biexponential parameters from noise-free data."""
        signal, f1_true, D1_true, D2_true = synthetic_biexp_single
        biexp_solver.fit(b_values, signal)
        params = biexp_solver.get_params()
        assert params["f1"] == pytest.approx(f1_true, rel=1e-2)
        assert params["D1"] == pytest.approx(D1_true, rel=1e-2)
        assert params["D2"] == pytest.approx(D2_true, rel=1e-2)

    def test_biexp_fit_diagnostics_pcov_shape(
        self, biexp_solver, b_values, synthetic_biexp_single
    ):
        """Covariance matrix in diagnostics has shape (n_params, n_params) for single voxel."""
        signal, *_ = synthetic_biexp_single
        biexp_solver.fit(b_values, signal)
        pcov = biexp_solver.get_diagnostics()["pcov"]
        n = biexp_solver.model.n_params
        assert pcov.shape == (n, n), f"Expected ({n}, {n}), got {pcov.shape}"

    @pytest.mark.parametrize(
        "fit_reduced, fit_s0, expected_names",
        [
            (True, False, ["f1", "D1", "D2"]),
            (False, False, ["f1", "D1", "f2", "D2"]),
            (True, True, ["f1", "D1", "D2", "S0"]),
        ],
        ids=["reduced", "full", "s0"],
    )
    def test_biexp_modes_param_count(self, fit_reduced, fit_s0, expected_names):
        """Solver stores correctly ordered param_names for each BiExpModel mode."""
        model = BiExpModel(fit_reduced=fit_reduced, fit_s0=fit_s0)
        p0 = {n: 0.1 for n in expected_names}
        bounds = {n: (0.0, 1.0) for n in expected_names}
        solver = CurveFitSolver(
            model=model, max_iter=100, tol=1e-6, p0=p0, bounds=bounds, method="trf"
        )
        assert solver.model.param_names == expected_names


# ---------------------------------------------------------------------------
# TestCurveFitSolverFitNoise
# ---------------------------------------------------------------------------


class TestCurveFitSolverFitNoise:
    """Tests for CurveFitSolver robustness with noisy data."""

    @pytest.mark.parametrize("noise_std", [0.5, 2.0, 5.0], ids=["low", "mid", "high"])
    def test_fit_noisy_data_within_tolerance(self, monoexp_solver, b_values, noise_std):
        """fit() recovers parameters within loose tolerance under Gaussian noise."""
        rng = np.random.default_rng(seed=42)
        S0_true, D_true = 1000.0, 0.001
        clean = MonoExpModel().forward(b_values, S0_true, D_true)
        noisy = clean + rng.normal(0, noise_std, size=clean.shape)
        monoexp_solver.fit(b_values, noisy)
        params = monoexp_solver.get_params()
        # Tolerance widens with noise — check order-of-magnitude correctness
        assert params["S0"] == pytest.approx(S0_true, rel=noise_std / S0_true * 20)
        assert params["D"] > 0, "Fitted D must be positive"

    def test_fit_identical_signal_across_voxels(self, monoexp_solver, b_values):
        """fit() handles N identical voxels and returns consistent per-voxel results."""
        S0_true, D_true = 1000.0, 0.001
        single = MonoExpModel().forward(b_values, S0_true, D_true)
        signals = np.tile(single, (5, 1))  # 5 identical voxels
        monoexp_solver.fit(b_values, signals)
        params = monoexp_solver.get_params()
        np.testing.assert_allclose(params["S0"], params["S0"][0], rtol=1e-6)
        np.testing.assert_allclose(params["D"], params["D"][0], rtol=1e-6)


# ---------------------------------------------------------------------------
# TestCurveFitSolverMethod
# ---------------------------------------------------------------------------


class TestCurveFitSolverMethod:
    """Tests for different scipy optimization methods."""

    @pytest.mark.parametrize("method", ["trf", "dogbox"], ids=["trf", "dogbox"])
    def test_fit_converges_with_different_methods(
        self, monoexp_model, monoexp_p0, monoexp_bounds, b_values, method
    ):
        """Both 'trf' and 'dogbox' methods converge to correct parameters."""
        solver = CurveFitSolver(
            model=monoexp_model,
            max_iter=2000,
            tol=1e-10,
            p0=monoexp_p0,
            bounds=monoexp_bounds,
            method=method,
        )
        S0_true, D_true = 1000.0, 0.001
        signal = MonoExpModel().forward(b_values, S0_true, D_true)
        solver.fit(b_values, signal)
        params = solver.get_params()
        assert params["S0"] == pytest.approx(S0_true, rel=1e-3)
        assert params["D"] == pytest.approx(D_true, rel=1e-3)

    def test_method_stored_correctly(self, monoexp_model, monoexp_p0, monoexp_bounds):
        """Solver stores the requested optimization method."""
        solver = CurveFitSolver(
            model=monoexp_model,
            max_iter=500,
            tol=1e-8,
            p0=monoexp_p0,
            bounds=monoexp_bounds,
            method="dogbox",
        )
        assert solver.method == "dogbox"


# ---------------------------------------------------------------------------
# TestCurveFitSolverFitData
# ---------------------------------------------------------------------------


class TestCurveFitSolverFitData:
    """Tests for CurveFitSolver._fit_data() output shapes and NaN propagation."""

    def test_popt_shape_single_pixel(self, monoexp_solver, b_values, synthetic_single):
        """_fit_data returns popt of shape (n_params, 1) for a single pixel."""
        signal, _, _ = synthetic_single
        ydata = signal[np.newaxis, :]
        p0 = np.array([[900.0], [0.0012]])
        bounds = (np.array([[0.0], [0.0]]), np.array([[5000.0], [0.05]]))
        popt, _ = monoexp_solver._fit_data(b_values, ydata, p0, bounds, n_pixels=1)
        assert popt.shape == (monoexp_solver.model.n_params, 1)

    def test_pcov_shape_single_pixel(self, monoexp_solver, b_values, synthetic_single):
        """_fit_data returns pcov of shape (1, n_params, n_params) for single pixel."""
        signal, _, _ = synthetic_single
        ydata = signal[np.newaxis, :]
        n = monoexp_solver.model.n_params
        p0 = np.array([[900.0], [0.0012]])
        bounds = (np.array([[0.0], [0.0]]), np.array([[5000.0], [0.05]]))
        _, pcov = monoexp_solver._fit_data(b_values, ydata, p0, bounds, n_pixels=1)
        assert pcov.shape == (1, n, n)

    def test_popt_shape_multi_pixel(self, monoexp_solver, b_values, synthetic_multi):
        """_fit_data returns popt of shape (n_params, n_pixels) for multiple pixels."""
        signals, _ = synthetic_multi
        n_pixels = signals.shape[0]
        n_params = monoexp_solver.model.n_params
        p0 = np.tile(np.array([900.0, 0.0012])[:, None], (1, n_pixels))
        bounds = (
            np.tile(np.array([0.0, 0.0])[:, None], (1, n_pixels)),
            np.tile(np.array([5000.0, 0.05])[:, None], (1, n_pixels)),
        )
        popt, _ = monoexp_solver._fit_data(b_values, signals, p0, bounds, n_pixels)
        assert popt.shape == (n_params, n_pixels)

    def test_pcov_shape_multi_pixel(self, monoexp_solver, b_values, synthetic_multi):
        """_fit_data returns pcov of shape (n_pixels, n_params, n_params) for multiple pixels."""
        signals, _ = synthetic_multi
        n_pixels = signals.shape[0]
        n = monoexp_solver.model.n_params
        p0 = np.tile(np.array([900.0, 0.0012])[:, None], (1, n_pixels))
        bounds = (
            np.tile(np.array([0.0, 0.0])[:, None], (1, n_pixels)),
            np.tile(np.array([5000.0, 0.05])[:, None], (1, n_pixels)),
        )
        _, pcov = monoexp_solver._fit_data(b_values, signals, p0, bounds, n_pixels)
        assert pcov.shape == (n_pixels, n, n)

    def test_failed_pixel_produces_nan_in_popt(
        self, monoexp_solver, b_values, synthetic_multi, mocker
    ):
        """When a single pixel fails, its popt column is NaN while others remain valid."""
        signals, _ = synthetic_multi
        n_pixels = signals.shape[0]
        call_count = {"n": 0}
        original = monoexp_solver._fit_single_pixel

        def selective_fail(xdata, ydata, p0, bounds, pixel_idx=None, pixel_fixed=None):
            call_count["n"] += 1
            if call_count["n"] == 2:  # fail only the second pixel
                return p0, np.full((len(p0), len(p0)), np.nan)
            return original(xdata, ydata, p0, bounds, pixel_idx=pixel_idx)

        mocker.patch.object(
            monoexp_solver, "_fit_single_pixel", side_effect=selective_fail
        )
        p0 = np.tile(np.array([900.0, 0.0012])[:, None], (1, n_pixels))
        bounds = (
            np.tile(np.array([0.0, 0.0])[:, None], (1, n_pixels)),
            np.tile(np.array([5000.0, 0.05])[:, None], (1, n_pixels)),
        )
        popt, pcov = monoexp_solver._fit_data(b_values, signals, p0, bounds, n_pixels)
        assert not np.any(np.isnan(popt[:, 0])), "Pixel 0 should have valid popt"
        assert not np.any(np.isnan(popt[:, 2])), "Pixel 2 should have valid popt"
        assert np.all(np.isnan(pcov[1])), "Pixel 1 covariance should be NaN"


# ---------------------------------------------------------------------------
# TestCurveFitSolverDiagnostics
# ---------------------------------------------------------------------------


class TestCurveFitSolverDiagnostics:
    """Tests for diagnostics content and get_params/get_diagnostics isolation."""

    def test_diagnostics_n_pixels_single(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """diagnostics_['n_pixels'] is 1 for single-voxel fit."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(b_values, signal)
        assert monoexp_solver.get_diagnostics()["n_pixels"] == 1

    def test_diagnostics_n_pixels_multi(
        self, monoexp_solver, b_values, synthetic_multi
    ):
        """diagnostics_['n_pixels'] equals the number of voxels for multi-voxel fit."""
        signals, param_sets = synthetic_multi
        monoexp_solver.fit(b_values, signals)
        assert monoexp_solver.get_diagnostics()["n_pixels"] == len(param_sets)

    def test_diagnostics_pcov_single_shape(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """diagnostics_['pcov'] is (n_params, n_params) for a single-voxel fit."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(b_values, signal)
        n = monoexp_solver.model.n_params
        pcov = monoexp_solver.get_diagnostics()["pcov"]
        assert pcov.shape == (n, n), f"Expected ({n}, {n}), got {pcov.shape}"

    def test_diagnostics_pcov_multi_shape(
        self, monoexp_solver, b_values, synthetic_multi
    ):
        """diagnostics_['pcov'] is (n_voxels, n_params, n_params) for multi-voxel fit."""
        signals, param_sets = synthetic_multi
        monoexp_solver.fit(b_values, signals)
        n = monoexp_solver.model.n_params
        pcov = monoexp_solver.get_diagnostics()["pcov"]
        assert pcov.shape == (len(param_sets), n, n)

    def test_get_params_returns_copy(self, monoexp_solver, b_values, synthetic_multi):
        """get_params() returns an independent copy — mutating it leaves params_ unchanged."""
        signals, _ = synthetic_multi
        monoexp_solver.fit(b_values, signals)
        params_copy = monoexp_solver.get_params()
        params_copy["S0"] = np.zeros(3)
        original_s0 = monoexp_solver.params_["S0"]
        assert not np.all(original_s0 == 0), "params_ should not be mutated"

    def test_get_diagnostics_returns_copy(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """get_diagnostics() returns an independent copy — mutating it leaves diagnostics_ unchanged."""
        signal, _, _ = synthetic_single
        monoexp_solver.fit(b_values, signal)
        diag_copy = monoexp_solver.get_diagnostics()
        diag_copy["n_pixels"] = 99
        assert monoexp_solver.diagnostics_["n_pixels"] == 1

    def test_fit_returns_self_for_chaining(
        self, monoexp_solver, b_values, synthetic_single
    ):
        """fit() returns the solver itself; params_ keys exactly match model.param_names."""
        signal, _, _ = synthetic_single
        returned = monoexp_solver.fit(b_values, signal)
        assert returned is monoexp_solver, (
            "fit() should return self for method chaining"
        )
        assert (
            list(monoexp_solver.get_params().keys()) == monoexp_solver.model.param_names
        )


# ---------------------------------------------------------------------------
# TestCurveFitSolverFixedParams
# ---------------------------------------------------------------------------


class TestCurveFitSolverFixedParams:
    """Tests for fixed-parameter support in CurveFitSolver."""

    @pytest.fixture
    def t1_model(self):
        """MonoExpModel with T1 correction enabled."""
        return MonoExpModel(fit_t1=True, repetition_time=3000.0)

    @pytest.fixture
    def t1_solver_fixed_scalar(self, t1_model):
        """Solver where T1 is fixed at model level (scalar)."""
        model = MonoExpModel(
            fit_t1=True, repetition_time=3000.0, fixed_params={"T1": 1000.0}
        )
        return CurveFitSolver(
            model=model,
            max_iter=2000,
            tol=1e-10,
            p0={"S0": 900.0, "D": 0.0012},
            bounds={"S0": (0.0, 5000.0), "D": (0.0, 0.05)},
            method="trf",
        )

    @pytest.fixture
    def t1_solver_no_fixed(self, t1_model):
        """Solver with T1 model but NO fixed params — for per-pixel testing.

        p0/bounds include T1 because it is in model.param_names.
        The per-pixel fixed values override T1 during fitting.
        """
        return CurveFitSolver(
            model=t1_model,
            max_iter=2000,
            tol=1e-10,
            p0={"S0": 900.0, "D": 0.0012, "T1": 1000.0},
            bounds={"S0": (0.0, 5000.0), "D": (0.0, 0.05), "T1": (100.0, 5000.0)},
            method="trf",
        )

    def _make_t1_signal(self, b_values, S0, D, T1, TR=3000.0):
        """Generate noise-free mono-exp signal with T1 correction."""
        model = MonoExpModel(fit_t1=True, repetition_time=TR)
        return model.forward(b_values, S0, D, T1)

    # --- scalar fixed ---

    @pytest.mark.unit
    def test_scalar_fixed_param_names(self, t1_solver_fixed_scalar):
        """Solver model param_names excludes the scalar-fixed T1."""
        assert t1_solver_fixed_scalar.model.param_names == ["S0", "D"]

    @pytest.mark.unit
    def test_scalar_fixed_recovers_params(self, b_values, t1_solver_fixed_scalar):
        """Solver with scalar fixed T1 recovers S0 and D from noise-free signal."""
        S0_true, D_true, T1_true = 1000.0, 0.001, 1000.0
        signal = self._make_t1_signal(b_values, S0_true, D_true, T1_true)
        t1_solver_fixed_scalar.fit(b_values, signal)
        params = t1_solver_fixed_scalar.get_params()
        assert set(params.keys()) == {"S0", "D"}
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)

    # --- per-pixel fixed ---

    @pytest.mark.unit
    def test_per_pixel_fixed_recovers_params(self, b_values, t1_solver_no_fixed):
        """Per-pixel T1 maps produce correct S0 and D for each voxel."""
        T1_values = np.array([800.0, 1000.0, 1500.0])
        S0_true, D_true = 1000.0, 0.001
        signals = np.array(
            [self._make_t1_signal(b_values, S0_true, D_true, T1) for T1 in T1_values]
        )
        t1_solver_no_fixed.fit(
            b_values,
            signals,
            pixel_fixed_params={"T1": T1_values},
        )
        params = t1_solver_no_fixed.get_params()
        assert set(params.keys()) == {"S0", "D"}
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)

    @pytest.mark.unit
    def test_per_pixel_fixed_multi_threaded(self, b_values, t1_model):
        """Per-pixel fixed params with multi_threading=True yields same results."""
        solver = CurveFitSolver(
            model=t1_model,
            max_iter=2000,
            tol=1e-10,
            p0={"S0": 900.0, "D": 0.0012, "T1": 1000.0},
            bounds={"S0": (0.0, 5000.0), "D": (0.0, 0.05), "T1": (100.0, 5000.0)},
            method="trf",
            multi_threading=True,
            n_pools=2,
        )
        T1_values = np.array([800.0, 1000.0, 1500.0])
        S0_true, D_true = 1000.0, 0.001
        signals = np.array(
            [self._make_t1_signal(b_values, S0_true, D_true, T1) for T1 in T1_values]
        )
        solver.fit(b_values, signals, pixel_fixed_params={"T1": T1_values})
        params = solver.get_params()
        np.testing.assert_allclose(params["S0"], S0_true, rtol=1e-2)
        np.testing.assert_allclose(params["D"], D_true, rtol=1e-2)
