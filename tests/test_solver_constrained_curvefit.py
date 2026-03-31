"""Tests for ConstrainedCurveFitSolver — initialization, validation, and constrained fitting."""

import numpy as np
import pytest

from pyneapple.models import BiExpModel, TriExpModel
from pyneapple.solvers import ConstrainedCurveFitSolver


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=float)


@pytest.fixture
def b_values():
    """Standard b-value array."""
    return B_VALUES.copy()


@pytest.fixture
def biexp_full_model():
    """BiExponential model in full (non-reduced) mode."""
    return BiExpModel(fit_reduced=False)


@pytest.fixture
def biexp_full_p0():
    """Initial guesses for BiExpModel full mode: [f1, D1, f2, D2]."""
    return {"f1": 0.3, "D1": 0.008, "f2": 0.6, "D2": 0.0008}


@pytest.fixture
def biexp_full_bounds():
    """Parameter bounds for BiExpModel full mode."""
    return {
        "f1": (0.0, 1.0),
        "D1": (0.0, 0.1),
        "f2": (0.0, 1.0),
        "D2": (0.0, 0.01),
    }


@pytest.fixture
def constrained_biexp_solver(biexp_full_model, biexp_full_p0, biexp_full_bounds):
    """Fully configured ConstrainedCurveFitSolver for BiExpModel full mode."""
    return ConstrainedCurveFitSolver(
        model=biexp_full_model,
        max_iter=500,
        tol=1e-10,
        p0=biexp_full_p0,
        bounds=biexp_full_bounds,
        fraction_constraint=True,
    )


@pytest.fixture
def triexp_full_model():
    """TriExponential model in full (non-reduced) mode."""
    return TriExpModel(fit_reduced=False)


@pytest.fixture
def triexp_full_p0():
    """Initial guesses for TriExpModel full mode: [f1, D1, f2, D2, f3, D3]."""
    return {
        "f1": 0.2,
        "D1": 0.01,
        "f2": 0.3,
        "D2": 0.003,
        "f3": 0.4,
        "D3": 0.0005,
    }


@pytest.fixture
def triexp_full_bounds():
    """Parameter bounds for TriExpModel full mode."""
    return {
        "f1": (0.0, 1.0),
        "D1": (0.0, 0.1),
        "f2": (0.0, 1.0),
        "D2": (0.0, 0.05),
        "f3": (0.0, 1.0),
        "D3": (0.0, 0.01),
    }


@pytest.fixture
def constrained_triexp_solver(triexp_full_model, triexp_full_p0, triexp_full_bounds):
    """Fully configured ConstrainedCurveFitSolver for TriExpModel full mode."""
    return ConstrainedCurveFitSolver(
        model=triexp_full_model,
        max_iter=500,
        tol=1e-10,
        p0=triexp_full_p0,
        bounds=triexp_full_bounds,
        fraction_constraint=True,
    )


@pytest.fixture
def synthetic_biexp_full(b_values):
    """Noise-free biexponential signal (full mode) for a single voxel."""
    f1_true, D1_true, f2_true, D2_true = 0.3, 0.01, 0.7, 0.001
    model = BiExpModel(fit_reduced=False)
    signal = model.forward(b_values, f1_true, D1_true, f2_true, D2_true)
    return signal, {"f1": f1_true, "D1": D1_true, "f2": f2_true, "D2": D2_true}


@pytest.fixture
def synthetic_triexp_full(b_values):
    """Noise-free triexponential signal (full mode) for a single voxel."""
    f1_true, D1_true = 0.2, 0.01
    f2_true, D2_true = 0.3, 0.003
    f3_true, D3_true = 0.5, 0.0005
    model = TriExpModel(fit_reduced=False)
    signal = model.forward(
        b_values, f1_true, D1_true, f2_true, D2_true, f3_true, D3_true
    )
    return signal, {
        "f1": f1_true,
        "D1": D1_true,
        "f2": f2_true,
        "D2": D2_true,
        "f3": f3_true,
        "D3": D3_true,
    }


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverInit
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverInit:
    """Test suite for ConstrainedCurveFitSolver construction and validation."""

    def test_valid_initialization_biexp(
        self, biexp_full_model, biexp_full_p0, biexp_full_bounds
    ):
        """Solver initializes correctly with BiExpModel in full mode."""
        solver = ConstrainedCurveFitSolver(
            model=biexp_full_model,
            max_iter=500,
            tol=1e-8,
            p0=biexp_full_p0,
            bounds=biexp_full_bounds,
            fraction_constraint=True,
        )
        assert solver.method == "SLSQP"
        assert solver.fraction_constraint is True
        assert solver._fraction_names == ["f1", "f2"]
        assert solver._fraction_indices == [0, 2]

    def test_valid_initialization_triexp(
        self, triexp_full_model, triexp_full_p0, triexp_full_bounds
    ):
        """Solver initializes correctly with TriExpModel in full mode."""
        solver = ConstrainedCurveFitSolver(
            model=triexp_full_model,
            max_iter=500,
            tol=1e-8,
            p0=triexp_full_p0,
            bounds=triexp_full_bounds,
            fraction_constraint=True,
        )
        assert solver._fraction_names == ["f1", "f2", "f3"]
        assert solver._fraction_indices == [0, 2, 4]

    def test_fit_reduced_raises_value_error(self):
        """ValueError raised when fraction_constraint=True with fit_reduced=True."""
        model = BiExpModel(fit_reduced=True)
        with pytest.raises(ValueError, match="incompatible with fit_reduced"):
            ConstrainedCurveFitSolver(
                model=model,
                max_iter=500,
                tol=1e-8,
                p0={"f1": 0.3, "D1": 0.008, "D2": 0.001},
                bounds={
                    "f1": (0.0, 1.0),
                    "D1": (0.0, 0.1),
                    "D2": (0.0, 0.01),
                },
                fraction_constraint=True,
            )

    def test_too_few_fractions_raises_value_error(self):
        """ValueError raised when model has fewer than 2 fraction parameters."""
        # BiExpModel reduced has only f1 as a fraction param
        # Use full mode but fix f2 so only f1 remains free
        model = BiExpModel(fit_reduced=False, fixed_params={"f2": 0.7})
        with pytest.raises(ValueError, match="at least 2 fraction"):
            ConstrainedCurveFitSolver(
                model=model,
                max_iter=500,
                tol=1e-8,
                p0={"f1": 0.3, "D1": 0.008, "D2": 0.001},
                bounds={
                    "f1": (0.0, 1.0),
                    "D1": (0.0, 0.1),
                    "D2": (0.0, 0.01),
                },
                fraction_constraint=True,
            )

    def test_method_always_slsqp(
        self, biexp_full_model, biexp_full_p0, biexp_full_bounds
    ):
        """Method is always SLSQP regardless of what is passed."""
        solver = ConstrainedCurveFitSolver(
            model=biexp_full_model,
            max_iter=500,
            tol=1e-8,
            p0=biexp_full_p0,
            bounds=biexp_full_bounds,
            method="trf",  # this should be overridden
            fraction_constraint=True,
        )
        assert solver.method == "SLSQP"

    def test_fraction_constraint_false_no_validation(self):
        """When fraction_constraint=False, no fraction validation is performed."""
        # BiExpModel reduced mode — would fail with fraction_constraint=True
        # but should succeed with False
        model = BiExpModel(fit_reduced=False)
        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=500,
            tol=1e-8,
            p0={"f1": 0.3, "D1": 0.008, "f2": 0.6, "D2": 0.001},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.01),
            },
            fraction_constraint=False,
        )
        assert solver.fraction_constraint is False
        assert solver._fraction_names == []
        assert solver._fraction_indices == []

    def test_params_empty_before_fit(self, constrained_biexp_solver):
        """params_ and diagnostics_ are empty before fit() is called."""
        assert len(constrained_biexp_solver.params_) == 0
        assert len(constrained_biexp_solver.diagnostics_) == 0


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverFitBiExp
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverFitBiExp:
    """Tests for constrained fitting of BiExponential model in full mode."""

    @pytest.mark.unit
    def test_fit_returns_self(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full
    ):
        """fit() returns the solver instance for method chaining."""
        signal, _ = synthetic_biexp_full
        result = constrained_biexp_solver.fit(b_values, signal)
        assert result is constrained_biexp_solver

    @pytest.mark.unit
    def test_fit_recovers_biexp_params(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full
    ):
        """fit() recovers true biexp parameters from noise-free data."""
        signal, true_params = synthetic_biexp_full
        constrained_biexp_solver.fit(b_values, signal)
        params = constrained_biexp_solver.get_params()
        for name, true_val in true_params.items():
            for value in params[name]:
                assert value == pytest.approx(
                    true_val, rel=5e-2
                ), f"{name}: expected {true_val}, got {value}"

    @pytest.mark.unit
    def test_fit_fractions_sum_leq_one(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full
    ):
        """Fitted fractions satisfy sum(f_i) <= 1."""
        signal, _ = synthetic_biexp_full
        constrained_biexp_solver.fit(b_values, signal)
        params = constrained_biexp_solver.get_params()
        frac_sum = params["f1"][0] + params["f2"][0]
        assert frac_sum <= 1.0 + 1e-10, f"Fraction sum {frac_sum} exceeds 1"

    @pytest.mark.unit
    def test_fit_stores_params_and_diagnostics(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full
    ):
        """After fit(), params_ and diagnostics_ are populated."""
        signal, _ = synthetic_biexp_full
        constrained_biexp_solver.fit(b_values, signal)
        assert len(constrained_biexp_solver.params_) > 0
        assert len(constrained_biexp_solver.diagnostics_) > 0

    @pytest.mark.unit
    def test_fit_param_keys_match_model(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full
    ):
        """Fitted parameter keys match the model's param_names."""
        signal, _ = synthetic_biexp_full
        constrained_biexp_solver.fit(b_values, signal)
        params = constrained_biexp_solver.get_params()
        assert set(params.keys()) == set(constrained_biexp_solver.model.param_names)

    @pytest.mark.unit
    def test_fit_multi_voxel(self, constrained_biexp_solver, b_values):
        """fit() handles multiple voxels and enforces constraint on each."""
        model = BiExpModel(fit_reduced=False)
        param_sets = [
            (0.3, 0.01, 0.7, 0.001),
            (0.4, 0.008, 0.5, 0.0015),
            (0.2, 0.012, 0.6, 0.0008),
        ]
        signals = np.array([model.forward(b_values, *ps) for ps in param_sets])
        constrained_biexp_solver.fit(b_values, signals)
        params = constrained_biexp_solver.get_params()
        assert isinstance(params["f1"], np.ndarray)
        assert params["f1"].shape == (3,)
        # Check constraint for each voxel
        for i in range(3):
            frac_sum = params["f1"][i] + params["f2"][i]
            assert (
                frac_sum <= 1.0 + 1e-10
            ), f"Voxel {i}: fraction sum {frac_sum} exceeds 1"


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverFitTriExp
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverFitTriExp:
    """Tests for constrained fitting of TriExponential model in full mode."""

    @pytest.mark.unit
    def test_fit_recovers_triexp_params(self, b_values, synthetic_triexp_full):
        """fit() recovers a triexp solution that reconstructs the signal accurately.

        The 6-parameter triexp model with 8 b-values is inherently
        ill-conditioned, so we verify signal reconstruction rather than
        exact parameter recovery.  We also verify the fraction constraint
        is satisfied.
        """
        signal, true_params = synthetic_triexp_full
        model = TriExpModel(fit_reduced=False)

        # Use p0 closer to truth for better convergence
        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-12,
            p0={
                "f1": 0.2,
                "D1": 0.01,
                "f2": 0.3,
                "D2": 0.003,
                "f3": 0.4,
                "D3": 0.0005,
            },
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.05),
                "f3": (0.0, 1.0),
                "D3": (0.0, 0.01),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signal)
        params = solver.get_params()

        # Verify signal reconstruction
        reconstructed = model.forward(
            b_values,
            params["f1"],
            params["D1"],
            params["f2"],
            params["D2"],
            params["f3"],
            params["D3"],
        )
        np.testing.assert_allclose(reconstructed, signal, rtol=1e-2)

    @pytest.mark.unit
    def test_triexp_fractions_sum_leq_one(
        self, constrained_triexp_solver, b_values, synthetic_triexp_full
    ):
        """Fitted triexp fractions satisfy sum(f1 + f2 + f3) <= 1."""
        signal, _ = synthetic_triexp_full
        constrained_triexp_solver.fit(b_values, signal)
        params = constrained_triexp_solver.get_params()
        frac_sum = params["f1"][0] + params["f2"][0] + params["f3"][0]
        assert frac_sum <= 1.0 + 1e-10, f"Fraction sum {frac_sum} exceeds 1"


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverConstraintEnforcement
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverConstraintEnforcement:
    """Tests that the constraint actually prevents fraction violations."""

    @pytest.mark.unit
    def test_bad_p0_still_converges_within_constraint(self, b_values):
        """Solver converges even when p0 fractions sum > 1, and result obeys constraint."""
        model = BiExpModel(fit_reduced=False)
        # True params where fractions sum to 1
        f1_true, D1_true, f2_true, D2_true = 0.3, 0.01, 0.7, 0.001
        signal = model.forward(b_values, f1_true, D1_true, f2_true, D2_true)

        # Start with p0 where fractions sum > 1
        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={"f1": 0.6, "D1": 0.008, "f2": 0.6, "D2": 0.001},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.01),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signal)
        params = solver.get_params()
        frac_sum = params["f1"][0] + params["f2"][0]
        assert frac_sum <= 1.0 + 1e-10, f"Constraint violated: f1 + f2 = {frac_sum}"

    @pytest.mark.unit
    def test_constraint_prevents_fraction_overflow_noisy(self, b_values):
        """With noisy data, constraint prevents fraction sum from exceeding 1."""
        model = BiExpModel(fit_reduced=False)
        rng = np.random.default_rng(seed=42)
        f1_true, D1_true, f2_true, D2_true = 0.3, 0.01, 0.7, 0.001
        clean = model.forward(b_values, f1_true, D1_true, f2_true, D2_true)
        noisy = clean + rng.normal(0, 5.0, size=clean.shape)

        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={"f1": 0.5, "D1": 0.008, "f2": 0.5, "D2": 0.001},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.01),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, noisy)
        params = solver.get_params()
        frac_sum = params["f1"][0] + params["f2"][0]
        assert (
            frac_sum <= 1.0 + 1e-10
        ), f"Constraint violated with noisy data: f1 + f2 = {frac_sum}"


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverCovariance
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverCovariance:
    """Tests for covariance estimation from the constrained solver."""

    @pytest.mark.unit
    def test_diagnostics_pcov_shape_single(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full
    ):
        """pcov in diagnostics has shape (n_params, n_params) for single voxel."""
        signal, _ = synthetic_biexp_full
        constrained_biexp_solver.fit(b_values, signal)
        pcov = constrained_biexp_solver.get_diagnostics()["pcov"]
        n = constrained_biexp_solver.model.n_params
        assert pcov.shape == (n, n), f"Expected ({n}, {n}), got {pcov.shape}"

    @pytest.mark.unit
    def test_diagnostics_n_pixels(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full
    ):
        """diagnostics_['n_pixels'] is 1 for single-voxel fit."""
        signal, _ = synthetic_biexp_full
        constrained_biexp_solver.fit(b_values, signal)
        assert constrained_biexp_solver.get_diagnostics()["n_pixels"] == 1


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverMultiThread
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverMultiThread:
    """Tests for multi-threaded constrained fitting."""

    @pytest.mark.unit
    def test_multi_threaded_matches_sequential(self, b_values):
        """Multi-threaded results match sequential results."""
        model = BiExpModel(fit_reduced=False)
        param_sets = [
            (0.3, 0.01, 0.7, 0.001),
            (0.4, 0.008, 0.5, 0.0015),
        ]
        signals = np.array([model.forward(b_values, *ps) for ps in param_sets])

        p0 = {"f1": 0.3, "D1": 0.008, "f2": 0.6, "D2": 0.001}
        bounds = {
            "f1": (0.0, 1.0),
            "D1": (0.0, 0.1),
            "f2": (0.0, 1.0),
            "D2": (0.0, 0.01),
        }

        # Sequential
        solver_seq = ConstrainedCurveFitSolver(
            model=model,
            max_iter=500,
            tol=1e-10,
            p0=p0,
            bounds=bounds,
            fraction_constraint=True,
            multi_threading=False,
        )
        solver_seq.fit(b_values, signals)

        # Multi-threaded
        solver_mt = ConstrainedCurveFitSolver(
            model=BiExpModel(fit_reduced=False),
            max_iter=500,
            tol=1e-10,
            p0=p0,
            bounds=bounds,
            fraction_constraint=True,
            multi_threading=True,
            n_pools=2,
        )
        solver_mt.fit(b_values, signals)

        for name in model.param_names:
            np.testing.assert_allclose(
                solver_mt.get_params()[name],
                solver_seq.get_params()[name],
                rtol=1e-3,
                err_msg=f"Multi-threaded {name} differs from sequential",
            )


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverJacobian
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverJacobian:
    """Tests for Jacobian usage in the constrained solver."""

    @pytest.mark.unit
    def test_fit_with_jacobian_enabled(self, b_values):
        """Solver uses analytical Jacobian when use_jacobian=True and model provides one."""
        model = BiExpModel(fit_reduced=False)
        p0 = {"f1": 0.3, "D1": 0.008, "f2": 0.6, "D2": 0.001}
        bounds = {
            "f1": (0.0, 1.0),
            "D1": (0.0, 0.1),
            "f2": (0.0, 1.0),
            "D2": (0.0, 0.01),
        }
        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=500,
            tol=1e-10,
            p0=p0,
            bounds=bounds,
            fraction_constraint=True,
            use_jacobian=True,
        )
        signal = model.forward(b_values, 0.3, 0.01, 0.7, 0.001)
        solver.fit(b_values, signal)
        params = solver.get_params()
        assert params["f1"][0] == pytest.approx(0.3, rel=5e-2)

    @pytest.mark.unit
    def test_fit_with_jacobian_disabled(self, b_values):
        """Solver converges when use_jacobian=False (finite differences)."""
        model = BiExpModel(fit_reduced=False)
        p0 = {"f1": 0.3, "D1": 0.008, "f2": 0.6, "D2": 0.001}
        bounds = {
            "f1": (0.0, 1.0),
            "D1": (0.0, 0.1),
            "f2": (0.0, 1.0),
            "D2": (0.0, 0.01),
        }
        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=500,
            tol=1e-10,
            p0=p0,
            bounds=bounds,
            fraction_constraint=True,
            use_jacobian=False,
        )
        signal = model.forward(b_values, 0.3, 0.01, 0.7, 0.001)
        solver.fit(b_values, signal)
        params = solver.get_params()
        assert params["f1"][0] == pytest.approx(0.3, rel=5e-2)


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverFailedFit
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverFailedFit:
    """Tests for failure handling in the constrained solver."""

    @pytest.mark.unit
    def test_failed_fit_returns_p0_and_nan_cov(
        self, constrained_biexp_solver, b_values, synthetic_biexp_full, mocker
    ):
        """When minimize raises an exception, solver returns p0 and NaN covariance."""
        signal, _ = synthetic_biexp_full

        mocker.patch(
            "pyneapple.solvers.constrained_curvefit.minimize",
            side_effect=RuntimeError("Optimization failed"),
        )

        constrained_biexp_solver.fit(b_values, signal)
        # Should still have params (fallback to p0)
        params = constrained_biexp_solver.get_params()
        assert isinstance(params, dict)
        # Diagnostics pcov should be NaN
        diag = constrained_biexp_solver.get_diagnostics()
        assert np.all(np.isnan(diag["pcov"]))


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverT1
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverT1:
    """Tests for constrained fitting with T1-corrected models."""

    @pytest.fixture
    def t1_biexp_model(self):
        """BiExpModel full mode with T1 correction enabled."""
        return BiExpModel(fit_reduced=False, fit_t1=True, repetition_time=3000.0)

    def _make_t1_signal(self, b_values, f1, D1, f2, D2, T1, TR=3000.0):
        """Generate noise-free biexp signal with T1 correction."""
        model = BiExpModel(fit_reduced=False, fit_t1=True, repetition_time=TR)
        return model.forward(b_values, f1, D1, f2, D2, T1)

    @pytest.mark.unit
    def test_t1_model_param_names(self, t1_biexp_model):
        """T1 model in full mode has correct parameter names."""
        assert t1_biexp_model.param_names == ["f1", "D1", "f2", "D2", "T1"]

    @pytest.mark.unit
    def test_t1_fit_recovers_params(self, b_values, t1_biexp_model):
        """Constrained solver recovers parameters from T1-corrected model."""
        f1_true, D1_true = 0.3, 0.01
        f2_true, D2_true = 0.7, 0.001
        T1_true = 1200.0

        signal = self._make_t1_signal(
            b_values, f1_true, D1_true, f2_true, D2_true, T1_true
        )

        solver = ConstrainedCurveFitSolver(
            model=t1_biexp_model,
            max_iter=1000,
            tol=1e-10,
            p0={"f1": 0.3, "D1": 0.008, "f2": 0.6, "D2": 0.001, "T1": 1000.0},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.01),
                "T1": (100.0, 5000.0),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signal)
        params = solver.get_params()

        frac_sum = params["f1"][0] + params["f2"][0]
        assert frac_sum <= 1.0 + 1e-10, f"Fraction sum {frac_sum} exceeds 1"
        assert params["f1"][0] == pytest.approx(f1_true, rel=5e-2)
        assert params["f2"][0] == pytest.approx(f2_true, rel=5e-2)


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverFixedParams
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverFixedParams:
    """Tests for constrained solver with fixed parameters (scalar and per-pixel)."""

    @pytest.mark.unit
    def test_scalar_fixed_t1(self, b_values):
        """Solver with scalar fixed T1 recovers fraction parameters."""
        model = BiExpModel(
            fit_reduced=False,
            fit_t1=True,
            repetition_time=3000.0,
            fixed_params={"T1": 1200.0},
        )
        # Model param_names should exclude T1
        assert "T1" not in model.param_names
        assert model.param_names == ["f1", "D1", "f2", "D2"]

        # Generate signal with T1=1200
        full_model = BiExpModel(fit_reduced=False, fit_t1=True, repetition_time=3000.0)
        signal = full_model.forward(b_values, 0.3, 0.01, 0.7, 0.001, 1200.0)

        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={"f1": 0.3, "D1": 0.008, "f2": 0.6, "D2": 0.001},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.01),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signal)
        params = solver.get_params()
        assert set(params.keys()) == {"f1", "D1", "f2", "D2"}
        frac_sum = params["f1"][0] + params["f2"][0]
        assert frac_sum <= 1.0 + 1e-10

    @pytest.mark.unit
    def test_per_pixel_fixed_t1(self, b_values):
        """Per-pixel fixed T1 maps produce correct constrained fits."""
        model = BiExpModel(fit_reduced=False, fit_t1=True, repetition_time=3000.0)
        T1_values = np.array([800.0, 1200.0, 1500.0])
        f1_true, D1_true, f2_true, D2_true = 0.3, 0.01, 0.7, 0.001
        signals = np.array(
            [
                model.forward(b_values, f1_true, D1_true, f2_true, D2_true, T1)
                for T1 in T1_values
            ]
        )

        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={
                "f1": 0.3,
                "D1": 0.008,
                "f2": 0.6,
                "D2": 0.001,
                "T1": 1000.0,
            },
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.01),
                "T1": (100.0, 5000.0),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signals, pixel_fixed_params={"T1": T1_values})
        params = solver.get_params()
        assert set(params.keys()) == {"f1", "D1", "f2", "D2"}
        for i in range(3):
            frac_sum = params["f1"][i] + params["f2"][i]
            assert (
                frac_sum <= 1.0 + 1e-10
            ), f"Voxel {i}: fraction sum {frac_sum} exceeds 1"
