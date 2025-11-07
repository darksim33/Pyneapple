import pytest

import numpy as np
from pyneapple.fitting.gpubridge import gpu_fitter


@pytest.mark.gpu
def test_import_gpufit():
    import pygpufit

    assert pygpufit is not None


@pytest.fixture
def gpufit():
    import pygpufit.gpufit as gpufit

    return gpufit


@pytest.mark.gpu
def test_cuda_check(gpufit):

    gpufit.cuda_available()
    assert True


@pytest.mark.gpu
def test_models_available(gpufit):
    models = [
        "MONOEXP",
        "MONOEXP_RED",
        "MONOEXP_T1",
        "MONOEXP_T1_STEAM",
        "BIEXP",
        "BIEXP_RED",
        "BIEXP_S0",
        "BIEXP_T1",
        "BIEXP_T1_STEAM",
        "BIEXP_S0_T1",
        "BIEXP_S0_T1_STEAM",
        "TRIEXP",
        "TRIEXP_RED",
        "TRIEXP_S0",
        "TRIEXP_T1",
        "TRIEXP_T1_STEAM",
        "TRIEXP_S0_T1",
        "TRIEXP_S0_T1_STEAM",
    ]
    for model in models:
        assert getattr(gpufit.ModelID, model, None) is not None


class TestGPUFitBasic:
    @pytest.mark.gpu
    def test_gpu_fit_mono(self, gpufit):
        # Known signal: S = S0 * exp(-b*D)
        b_values = np.array([0, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        S0_true, D_true = 200.0, 0.0015

        # Generate known signal
        signal = S0_true * np.exp(-b_values * D_true)
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [210, 0.0015]
        lower = [10, 0.0007]
        upper = [2500, 0.003]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (2, 1))
        )

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.MONOEXP,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=250,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(result[0][0], [S0_true, D_true], rtol=0.001)

    @pytest.mark.gpu
    def test_gpu_fit_mono_red(self, gpufit):
        # Known signal (reduced): S = exp(-b*D)
        b_values = np.array([0, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        D_true = 0.0015

        # Generate known signal (normalized, no S0)
        signal = np.exp(-b_values * D_true)
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [0.0015]
        lower = [0.0007]
        upper = [0.003]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (1, 1))

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.MONOEXP_RED,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=250,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(result[0][0], [D_true], rtol=0.001)

    @pytest.mark.gpu
    def test_gpu_fit_biexp(self, gpufit):
        # Known signal: S = S0_tissue * exp(-b*D) + S0_perfusion * exp(-b*Dp)
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        S0_tissue_true, D_true, S0_perfusion_true, Dp_true = 150.0, 0.001, 50.0, 0.02

        # Generate known signal
        signal = S0_tissue_true * np.exp(
            -b_values * D_true
        ) + S0_perfusion_true * np.exp(-b_values * Dp_true)
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [150, 0.001, 50, 0.02]
        lower = [10, 0.0005, 10, 0.005]
        upper = [2500, 0.003, 2500, 0.05]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (4, 1))
        )

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.BIEXP,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=350,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(
            result[0][0],
            [S0_tissue_true, D_true, S0_perfusion_true, Dp_true],
            rtol=0.01,
        )

    @pytest.mark.gpu
    def test_gpu_fit_biexp_red(self, gpufit):
        # Known signal (reduced): S = (1-fp) * exp(-b*D) + fp * exp(-b*Dp)
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        f_true, D_true, Dp_true = 0.25, 0.001, 0.02

        # Generate known signal (normalized)
        signal = (f_true) * np.exp(-b_values * D_true) + (1 - f_true) * np.exp(
            -b_values * Dp_true
        )
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [0.25, 0.001, 0.02]
        lower = [0.05, 0.0005, 0.005]
        upper = [0.5, 0.003, 0.05]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (3, 1))
        )

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.BIEXP_RED,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=250,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(result[0][0], [f_true, D_true, Dp_true], rtol=0.01)

    @pytest.mark.gpu
    def test_gpu_fit_biexp_s0(self, gpufit):
        # Known signal: S = S0 * ((1-fp) * exp(-b*D) + fp * exp(-b*Dp))
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        f_true, D_true, Dp_true, S0_true = 0.25, 0.001, 0.02, 200.0

        # Generate known signal
        signal = (
            (f_true) * np.exp(-b_values * D_true)
            + (1 - f_true) * np.exp(-b_values * Dp_true)
        ) * S0_true
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [0.25, 0.001, 0.02, 210]
        lower = [0.05, 0.0005, 0.005, 10]
        upper = [0.5, 0.003, 0.05, 2500]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (4, 1))
        )

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.BIEXP_S0,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=250,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(result[0][0], [f_true, D_true, Dp_true, S0_true], rtol=0.01)

    @pytest.mark.gpu
    def test_gpu_fit_triexp(self, gpufit):
        # Known signal: S = S0_tissue * exp(-b*D) + S0_perfusion * exp(-b*Dp) + S0_vascular * exp(-b*Dv)
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        (
            S0_tissue_true,
            D_true,
            S0_perfusion_true,
            Dp_true,
            S0_vascular_true,
            Dv_true,
        ) = (
            100.0,
            0.001,
            50.0,
            0.02,
            50.0,
            0.1,
        )

        # Generate known signal
        signal = (
            S0_tissue_true * np.exp(-b_values * D_true)
            + S0_perfusion_true * np.exp(-b_values * Dp_true)
            + S0_vascular_true * np.exp(-b_values * Dv_true)
        )
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [100, 0.001, 50, 0.02, 50, 0.1]
        lower = [10, 0.0005, 10, 0.005, 10, 0.05]
        upper = [2500, 0.003, 2500, 0.05, 2500, 0.3]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (6, 1))
        )

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.TRIEXP,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=350,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(
            result[0][0],
            [
                S0_tissue_true,
                D_true,
                S0_perfusion_true,
                Dp_true,
                S0_vascular_true,
                Dv_true,
            ],
            rtol=0.01,
        )

    @pytest.mark.gpu
    def test_gpu_fit_triexp_red(self, gpufit):
        # Known signal (reduced): S = f1 * exp(-b*D) + f2 * exp(-b*Dp) + (1-f1-f2) * exp(-b*Dv)
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        f1_true, D_true, f2_true, Dp_true, Dv_true = 0.5, 0.001, 0.25, 0.02, 0.1

        # Generate known signal (normalized)
        signal = (
            f1_true * np.exp(-b_values * D_true)
            + f2_true * np.exp(-b_values * Dp_true)
            + (1 - f1_true - f2_true) * np.exp(-b_values * Dv_true)
        )
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [0.5, 0.001, 0.25, 0.02, 0.1]
        lower = [0.1, 0.0005, 0.05, 0.005, 0.05]
        upper = [0.7, 0.003, 0.5, 0.05, 0.3]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (5, 1))
        )

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.TRIEXP_RED,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=350,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(
            result[0][0], [f1_true, D_true, f2_true, Dp_true, Dv_true], rtol=0.01
        )

    @pytest.mark.gpu
    def test_gpu_fit_triexp_s0(self, gpufit):
        # Known signal: S = S0 * (f1 * exp(-b*D) + f2 * exp(-b*Dp) + (1-f1-f2) * exp(-b*Dv))
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float32)

        # Ground truth parameters
        f1_true, D_true, f2_true, Dp_true, Dv_true, S0_true = (
            0.5,
            0.001,
            0.25,
            0.02,
            0.1,
            200.0,
        )

        # Generate known signal
        signal = S0_true * (
            f1_true * np.exp(-b_values * D_true)
            + f2_true * np.exp(-b_values * Dp_true)
            + (1 - f1_true - f2_true) * np.exp(-b_values * Dv_true)
        )
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [0.55, 0.002, 0.3, 0.025, 0.1, 210]
        lower = [0.1, 0.0005, 0.05, 0.005, 0.05, 10]
        upper = [0.7, 0.003, 0.5, 0.05, 0.3, 2500]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (6, 1))
        )

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.TRIEXP_S0,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=350,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=b_values,
        )

        assert np.allclose(
            result[0][0],
            [f1_true, D_true, f2_true, Dp_true, Dv_true, S0_true],
            rtol=0.01,
        )


class TestGPUFitT1:
    @pytest.mark.gpu
    @pytest.mark.skip("Not working properly atm.")
    def test_gpu_fit_mono_t1(self, gpufit):
        # Known signal with T1: S = S0 * (1 - exp(-TR/T1)) * exp(-b*D)
        b_values = np.array([0, 100, 200, 400, 600, 800, 1000], dtype=np.float32)
        TR = 3000.0  # ms

        # Ground truth parameters
        S0_true, D_true, T1_true = 200.0, 0.0015, 1000.0

        # Generate known signal
        signal = (1 - np.exp(-TR / T1_true)) * np.exp(-b_values * D_true) * S0_true
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [180, 0.0015, 900]
        lower = [10, 0.0007, 500]
        upper = [2500, 0.003, 3000]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (3, 1))
        )
        user_info = np.append(b_values, np.float32(TR))

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.MONOEXP_T1,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=250,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=user_info,
        )

        assert np.allclose(result[0][0], [S0_true, D_true, T1_true], rtol=0.01)

    @pytest.mark.gpu
    @pytest.mark.skip("Not working properly atm.")
    def test_gpu_fit_biexp_t1(self, gpufit):
        # Known signal with T1: S = (S0_tissue * exp(-b*D) + S0_perfusion * exp(-b*Dp)) * (1 - exp(-TR/T1))
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float32)
        TR = 3000.0  # ms

        # Ground truth parameters
        S0_tissue_true, D_true, S0_perfusion_true, Dp_true, T1_true = (
            150.0,
            0.001,
            50.0,
            0.02,
            1200.0,
        )

        # Generate known signal
        signal = (
            S0_tissue_true * np.exp(-b_values * D_true)
            + S0_perfusion_true * np.exp(-b_values * Dp_true)
        ) * (1 - np.exp(-TR / T1_true))
        fit_data = signal.reshape(1, -1).astype(np.float32)

        starts = [160, 0.002, 70, 0.009, 1500]
        lower = [10, 0.0005, 10, 0.005, 500]
        upper = [2500, 0.003, 2500, 0.05, 3000]
        start_values = np.tile(np.float32(starts), (1, 1))
        constraints = np.tile(
            np.float32(list(zip(lower, upper))).flatten(),
            (1, 1),
        )

        constraint_types = np.squeeze(
            np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (5, 1))
        )
        user_info = np.append(b_values, np.float32(TR))

        result = gpufit.fit_constrained(
            fit_data,
            None,
            gpufit.ModelID.BIEXP_T1,
            initial_parameters=start_values,
            constraints=constraints,
            constraint_types=constraint_types,
            tolerance=1e-7,
            max_number_iterations=650,
            parameters_to_fit=None,
            estimator_id=gpufit.EstimatorID.LSE,
            user_info=user_info,
        )

        assert np.allclose(
            result[0][0],
            [S0_tissue_true, D_true, S0_perfusion_true, Dp_true, T1_true],
            rtol=0.05,
        )
