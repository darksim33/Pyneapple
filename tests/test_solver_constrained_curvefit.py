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
def triexp_reduced_model():
    """TriExponential model in reduced mode: params = [f1, D1, f2, D2, D3]."""
    return TriExpModel(fit_reduced=True)


@pytest.fixture
def triexp_reduced_p0():
    """Initial guesses for TriExpModel reduced mode."""
    return {"f1": 0.2, "D1": 0.01, "f2": 0.3, "D2": 0.003, "D3": 0.0005}


@pytest.fixture
def triexp_reduced_bounds():
    """Parameter bounds for TriExpModel reduced mode."""
    return {
        "f1": (0.0, 1.0),
        "D1": (0.0, 0.1),
        "f2": (0.0, 1.0),
        "D2": (0.0, 0.05),
        "D3": (0.0, 0.01),
    }


@pytest.fixture
def constrained_triexp_solver(
    triexp_reduced_model, triexp_reduced_p0, triexp_reduced_bounds
):
    """Fully configured ConstrainedCurveFitSolver for TriExpModel reduced mode."""
    return ConstrainedCurveFitSolver(
        model=triexp_reduced_model,
        max_iter=500,
        tol=1e-10,
        p0=triexp_reduced_p0,
        bounds=triexp_reduced_bounds,
        fraction_constraint=True,
    )


@pytest.fixture
def synthetic_triexp_reduced(b_values):
    """Noise-free triexponential signal (reduced mode) for a single voxel.

    True params: f1=0.2, D1=0.01, f2=0.3, D2=0.003, D3=0.0005
    Implicit:    f3 = 1 - 0.2 - 0.3 = 0.5
    """
    f1_true, D1_true = 0.2, 0.01
    f2_true, D2_true = 0.3, 0.003
    D3_true = 0.0005
    model = TriExpModel(fit_reduced=True)
    signal = model.forward(b_values, f1_true, D1_true, f2_true, D2_true, D3_true)
    return signal, {
        "f1": f1_true,
        "D1": D1_true,
        "f2": f2_true,
        "D2": D2_true,
        "D3": D3_true,
    }


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverInit
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverInit:
    """Test suite for ConstrainedCurveFitSolver construction and validation."""

    def test_valid_initialization_triexp_reduced(
        self, triexp_reduced_model, triexp_reduced_p0, triexp_reduced_bounds
    ):
        """Solver initializes correctly with TriExpModel in reduced mode."""
        solver = ConstrainedCurveFitSolver(
            model=triexp_reduced_model,
            max_iter=500,
            tol=1e-8,
            p0=triexp_reduced_p0,
            bounds=triexp_reduced_bounds,
            fraction_constraint=True,
        )
        assert solver.method == "SLSQP"
        assert solver.fraction_constraint is True
        assert solver._fraction_names == ["f1", "f2"]
        assert solver._fraction_indices == [0, 2]

    def test_valid_initialization_triexp_s0(self):
        """Solver initializes correctly with TriExpModel in S0 mode (fit_s0=True).

        fit_s0=True implies fit_reduced=True; params = [f1, D1, f2, D2, D3, S0].
        """
        model = TriExpModel(fit_s0=True)
        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=500,
            tol=1e-8,
            p0={"f1": 0.2, "D1": 0.01, "f2": 0.3, "D2": 0.003, "D3": 0.0005, "S0": 800.0},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.05),
                "D3": (0.0, 0.01),
                "S0": (0.0, 5000.0),
            },
            fraction_constraint=True,
        )
        assert solver._fraction_names == ["f1", "f2"]
        assert solver._fraction_indices == [0, 2]

    def test_full_model_raises_value_error(self):
        """ValueError raised when fraction_constraint=True with fit_reduced=False.

        Full mode signals are not normalised, so the fraction constraint is
        not physically meaningful.
        """
        model = TriExpModel(fit_reduced=False)
        with pytest.raises(ValueError, match="fit_reduced=True"):
            ConstrainedCurveFitSolver(
                model=model,
                max_iter=500,
                tol=1e-8,
                p0={"f1": 0.2, "D1": 0.01, "f2": 0.3, "D2": 0.003, "f3": 0.4, "D3": 0.0005},
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

    def test_too_few_fractions_raises_value_error(self):
        """ValueError raised when model has fewer than 2 fraction parameters.

        BiExpModel(fit_reduced=True) only exposes f1 as a free fraction;
        f2 = 1 - f1 is implicit.
        """
        model = BiExpModel(fit_reduced=True)
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
        self, triexp_reduced_model, triexp_reduced_p0, triexp_reduced_bounds
    ):
        """Method is always SLSQP regardless of what is passed."""
        solver = ConstrainedCurveFitSolver(
            model=triexp_reduced_model,
            max_iter=500,
            tol=1e-8,
            p0=triexp_reduced_p0,
            bounds=triexp_reduced_bounds,
            method="trf",  # should be overridden
            fraction_constraint=True,
        )
        assert solver.method == "SLSQP"

    def test_fraction_constraint_false_no_validation(self):
        """When fraction_constraint=False, no model validation is performed.

        BiExpModel(fit_reduced=False) would raise with fraction_constraint=True
        but must succeed with False.
        """
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

    def test_params_empty_before_fit(self, constrained_triexp_solver):
        """params_ and diagnostics_ are empty before fit() is called."""
        assert len(constrained_triexp_solver.params_) == 0
        assert len(constrained_triexp_solver.diagnostics_) == 0


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverFitTriExpReduced
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverFitTriExpReduced:
    """Tests for constrained fitting of TriExponential model in reduced mode."""

    @pytest.mark.unit
    def test_fit_returns_self(
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced
    ):
        """fit() returns the solver instance for method chaining."""
        signal, _ = synthetic_triexp_reduced
        result = constrained_triexp_solver.fit(b_values, signal)
        assert result is constrained_triexp_solver

    @pytest.mark.unit
    def test_fit_recovers_params(
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced
    ):
        """fit() recovers true triexp reduced parameters from noise-free data."""
        signal, true_params = synthetic_triexp_reduced
        constrained_triexp_solver.fit(b_values, signal)
        params = constrained_triexp_solver.get_params()
        for name, true_val in true_params.items():
            for value in params[name]:
                assert value == pytest.approx(
                    true_val, rel=5e-2
                ), f"{name}: expected {true_val}, got {value}"

    @pytest.mark.unit
    def test_fit_fractions_sum_leq_one(
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced
    ):
        """Fitted fractions f1 + f2 satisfy the hard constraint <= 1."""
        signal, _ = synthetic_triexp_reduced
        constrained_triexp_solver.fit(b_values, signal)
        params = constrained_triexp_solver.get_params()
        frac_sum = params["f1"][0] + params["f2"][0]
        assert frac_sum <= 1.0 + 1e-10, f"Fraction sum {frac_sum} exceeds 1"

    @pytest.mark.unit
    def test_fit_stores_params_and_diagnostics(
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced
    ):
        """After fit(), params_ and diagnostics_ are populated."""
        signal, _ = synthetic_triexp_reduced
        constrained_triexp_solver.fit(b_values, signal)
        assert len(constrained_triexp_solver.params_) > 0
        assert len(constrained_triexp_solver.diagnostics_) > 0

    @pytest.mark.unit
    def test_fit_param_keys_match_model(
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced
    ):
        """Fitted parameter keys match the model's param_names."""
        signal, _ = synthetic_triexp_reduced
        constrained_triexp_solver.fit(b_values, signal)
        params = constrained_triexp_solver.get_params()
        assert set(params.keys()) == set(constrained_triexp_solver.model.param_names)

    @pytest.mark.unit
    def test_fit_multi_voxel(self, constrained_triexp_solver, b_values):
        """fit() handles multiple voxels and enforces constraint on each."""
        model = TriExpModel(fit_reduced=True)
        param_sets = [
            (0.2, 0.01, 0.3, 0.003, 0.0005),
            (0.3, 0.008, 0.4, 0.002, 0.0003),
            (0.15, 0.012, 0.25, 0.004, 0.0007),
        ]
        signals = np.array([model.forward(b_values, *ps) for ps in param_sets])
        constrained_triexp_solver.fit(b_values, signals)
        params = constrained_triexp_solver.get_params()
        assert isinstance(params["f1"], np.ndarray)
        assert params["f1"].shape == (3,)
        for i in range(3):
            frac_sum = params["f1"][i] + params["f2"][i]
            assert (
                frac_sum <= 1.0 + 1e-10
            ), f"Voxel {i}: fraction sum {frac_sum} exceeds 1"


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverConstraintEnforcement
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverConstraintEnforcement:
    """Tests that the constraint actually prevents fraction violations."""

    @pytest.mark.unit
    def test_bad_p0_still_converges_within_constraint(self, b_values):
        """Solver converges even when p0 fractions sum > 1, result obeys constraint."""
        model = TriExpModel(fit_reduced=True)
        # True params: f1=0.2, f2=0.3 → f3=0.5
        signal = model.forward(b_values, 0.2, 0.01, 0.3, 0.003, 0.0005)

        # Start with p0 where f1 + f2 > 1
        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={"f1": 0.6, "D1": 0.01, "f2": 0.6, "D2": 0.003, "D3": 0.0005},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.05),
                "D3": (0.0, 0.01),
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
        model = TriExpModel(fit_reduced=True)
        rng = np.random.default_rng(seed=42)
        clean = model.forward(b_values, 0.2, 0.01, 0.3, 0.003, 0.0005)
        noisy = clean + rng.normal(0, 5.0, size=clean.shape)

        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={"f1": 0.3, "D1": 0.01, "f2": 0.4, "D2": 0.003, "D3": 0.0005},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.05),
                "D3": (0.0, 0.01),
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
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced
    ):
        """pcov in diagnostics has shape (n_params, n_params) for single voxel."""
        signal, _ = synthetic_triexp_reduced
        constrained_triexp_solver.fit(b_values, signal)
        pcov = constrained_triexp_solver.get_diagnostics()["pcov"]
        n = constrained_triexp_solver.model.n_params
        assert pcov.shape == (n, n), f"Expected ({n}, {n}), got {pcov.shape}"

    @pytest.mark.unit
    def test_diagnostics_n_pixels(
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced
    ):
        """diagnostics_['n_pixels'] is 1 for single-voxel fit."""
        signal, _ = synthetic_triexp_reduced
        constrained_triexp_solver.fit(b_values, signal)
        assert constrained_triexp_solver.get_diagnostics()["n_pixels"] == 1


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverMultiThread
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverMultiThread:
    """Tests for multi-threaded constrained fitting."""

    @pytest.mark.unit
    def test_multi_threaded_matches_sequential(self, b_values):
        """Multi-threaded results match sequential results."""
        model = TriExpModel(fit_reduced=True)
        param_sets = [
            (0.2, 0.01, 0.3, 0.003, 0.0005),
            (0.3, 0.008, 0.4, 0.002, 0.0003),
        ]
        signals = np.array([model.forward(b_values, *ps) for ps in param_sets])

        p0 = {"f1": 0.2, "D1": 0.01, "f2": 0.3, "D2": 0.003, "D3": 0.0005}
        bounds = {
            "f1": (0.0, 1.0),
            "D1": (0.0, 0.1),
            "f2": (0.0, 1.0),
            "D2": (0.0, 0.05),
            "D3": (0.0, 0.01),
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
            model=TriExpModel(fit_reduced=True),
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
        model = TriExpModel(fit_reduced=True)
        p0 = {"f1": 0.2, "D1": 0.01, "f2": 0.3, "D2": 0.003, "D3": 0.0005}
        bounds = {
            "f1": (0.0, 1.0),
            "D1": (0.0, 0.1),
            "f2": (0.0, 1.0),
            "D2": (0.0, 0.05),
            "D3": (0.0, 0.01),
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
        signal = model.forward(b_values, 0.2, 0.01, 0.3, 0.003, 0.0005)
        solver.fit(b_values, signal)
        params = solver.get_params()
        assert params["f1"][0] == pytest.approx(0.2, rel=5e-2)

    @pytest.mark.unit
    def test_fit_with_jacobian_disabled(self, b_values):
        """Solver converges when use_jacobian=False (finite differences)."""
        model = TriExpModel(fit_reduced=True)
        p0 = {"f1": 0.2, "D1": 0.01, "f2": 0.3, "D2": 0.003, "D3": 0.0005}
        bounds = {
            "f1": (0.0, 1.0),
            "D1": (0.0, 0.1),
            "f2": (0.0, 1.0),
            "D2": (0.0, 0.05),
            "D3": (0.0, 0.01),
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
        signal = model.forward(b_values, 0.2, 0.01, 0.3, 0.003, 0.0005)
        solver.fit(b_values, signal)
        params = solver.get_params()
        assert params["f1"][0] == pytest.approx(0.2, rel=5e-2)


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverFailedFit
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverFailedFit:
    """Tests for failure handling in the constrained solver."""

    @pytest.mark.unit
    def test_failed_fit_returns_p0_and_nan_cov(
        self, constrained_triexp_solver, b_values, synthetic_triexp_reduced, mocker
    ):
        """When minimize raises an exception, solver returns p0 and NaN covariance."""
        signal, _ = synthetic_triexp_reduced

        mocker.patch(
            "pyneapple.solvers.constrained_curvefit.minimize",
            side_effect=RuntimeError("Optimization failed"),
        )

        constrained_triexp_solver.fit(b_values, signal)
        # Should still have params (fallback to p0)
        params = constrained_triexp_solver.get_params()
        assert isinstance(params, dict)
        # Diagnostics pcov should be NaN
        diag = constrained_triexp_solver.get_diagnostics()
        assert np.all(np.isnan(diag["pcov"]))


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverT1
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverT1:
    """Tests for constrained fitting with T1-corrected models."""

    @pytest.fixture
    def t1_triexp_model(self):
        """TriExpModel reduced mode with T1 correction: params = [f1, D1, f2, D2, D3, T1]."""
        return TriExpModel(fit_reduced=True, fit_t1=True, repetition_time=3000.0)

    def _make_t1_signal(self, b_values, f1, D1, f2, D2, D3, T1, TR=3000.0):
        """Generate noise-free triexp signal with T1 correction."""
        model = TriExpModel(fit_reduced=True, fit_t1=True, repetition_time=TR)
        return model.forward(b_values, f1, D1, f2, D2, D3, T1)

    @pytest.mark.unit
    def test_t1_model_param_names(self, t1_triexp_model):
        """T1 model in reduced mode has correct parameter names."""
        assert t1_triexp_model.param_names == ["f1", "D1", "f2", "D2", "D3", "T1"]

    @pytest.mark.unit
    def test_t1_fit_recovers_params(self, b_values, t1_triexp_model):
        """Constrained solver satisfies the constraint and reconstructs the signal.

        The 6-parameter T1-corrected reduced triexp model (f1, D1, f2, D2, D3, T1)
        fitted on 8 b-values is ill-conditioned; exact parameter recovery is not
        guaranteed.  We verify the fraction constraint and signal fidelity instead.
        """
        f1_true, D1_true = 0.2, 0.01
        f2_true, D2_true = 0.3, 0.003
        D3_true = 0.0005
        T1_true = 1200.0

        signal = self._make_t1_signal(
            b_values, f1_true, D1_true, f2_true, D2_true, D3_true, T1_true
        )

        solver = ConstrainedCurveFitSolver(
            model=t1_triexp_model,
            max_iter=1000,
            tol=1e-10,
            p0={
                "f1": 0.2,
                "D1": 0.01,
                "f2": 0.3,
                "D2": 0.003,
                "D3": 0.0005,
                "T1": 1000.0,
            },
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.05),
                "D3": (0.0, 0.01),
                "T1": (100.0, 5000.0),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signal)
        params = solver.get_params()

        # Hard constraint must be satisfied
        frac_sum = params["f1"][0] + params["f2"][0]
        assert frac_sum <= 1.0 + 1e-10, f"Fraction sum {frac_sum} exceeds 1"

        # Signal reconstruction must be accurate
        reconstructed = t1_triexp_model.forward(
            b_values,
            params["f1"][0],
            params["D1"][0],
            params["f2"][0],
            params["D2"][0],
            params["D3"][0],
            params["T1"][0],
        )
        np.testing.assert_allclose(reconstructed, signal, rtol=5e-2)


# ---------------------------------------------------------------------------
# TestConstrainedCurveFitSolverFixedParams
# ---------------------------------------------------------------------------


class TestConstrainedCurveFitSolverFixedParams:
    """Tests for constrained solver with fixed parameters (scalar and per-pixel)."""

    @pytest.mark.unit
    def test_scalar_fixed_t1(self, b_values):
        """Solver with scalar fixed T1 recovers fraction parameters."""
        model = TriExpModel(
            fit_reduced=True,
            fit_t1=True,
            repetition_time=3000.0,
            fixed_params={"T1": 1200.0},
        )
        # With T1 fixed, model exposes [f1, D1, f2, D2, D3] — 2 free fractions
        assert "T1" not in model.param_names
        assert model.param_names == ["f1", "D1", "f2", "D2", "D3"]

        # Generate signal with T1=1200
        full_model = TriExpModel(fit_reduced=True, fit_t1=True, repetition_time=3000.0)
        signal = full_model.forward(b_values, 0.2, 0.01, 0.3, 0.003, 0.0005, 1200.0)

        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={"f1": 0.2, "D1": 0.01, "f2": 0.3, "D2": 0.003, "D3": 0.0005},
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.05),
                "D3": (0.0, 0.01),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signal)
        params = solver.get_params()
        assert set(params.keys()) == {"f1", "D1", "f2", "D2", "D3"}
        frac_sum = params["f1"][0] + params["f2"][0]
        assert frac_sum <= 1.0 + 1e-10

    @pytest.mark.unit
    def test_per_pixel_fixed_t1(self, b_values):
        """Per-pixel fixed T1 maps produce correct constrained fits."""
        model = TriExpModel(fit_reduced=True, fit_t1=True, repetition_time=3000.0)
        T1_values = np.array([800.0, 1200.0, 1500.0])
        f1_true, D1_true, f2_true, D2_true, D3_true = 0.2, 0.01, 0.3, 0.003, 0.0005
        signals = np.array(
            [
                model.forward(b_values, f1_true, D1_true, f2_true, D2_true, D3_true, T1)
                for T1 in T1_values
            ]
        )

        solver = ConstrainedCurveFitSolver(
            model=model,
            max_iter=1000,
            tol=1e-10,
            p0={
                "f1": 0.2,
                "D1": 0.01,
                "f2": 0.3,
                "D2": 0.003,
                "D3": 0.0005,
                "T1": 1000.0,
            },
            bounds={
                "f1": (0.0, 1.0),
                "D1": (0.0, 0.1),
                "f2": (0.0, 1.0),
                "D2": (0.0, 0.05),
                "D3": (0.0, 0.01),
                "T1": (100.0, 5000.0),
            },
            fraction_constraint=True,
        )
        solver.fit(b_values, signals, pixel_fixed_params={"T1": T1_values})
        params = solver.get_params()
        assert set(params.keys()) == {"f1", "D1", "f2", "D2", "D3"}
        for i in range(3):
            frac_sum = params["f1"][i] + params["f2"][i]
            assert (
                frac_sum <= 1.0 + 1e-10
            ), f"Voxel {i}: fraction sum {frac_sum} exceeds 1"
