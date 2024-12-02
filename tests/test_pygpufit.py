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
        "MONO_EXP",
        "MONO_EXP_RED",
        "BI_EXP",
        "BI_EXP_RED",
        "TRI_EXP",
        "TRI_EXP_RED",
    ]
    for model in models:
        assert getattr(gpufit.ModelID, model, None) is not None


@pytest.mark.gpu
def test_gpu_fit_mono(gpufit, ivim_mono_params):
    n_fits = 100
    ivim_mono_params.scale_image = "S/S0"
    fit_args = np.zeros((n_fits, ivim_mono_params.b_values.shape[0]), np.float32)
    # weights = np.ones((n_fits, ivim_mono_params.b_values.shape[0]), np.float32)
    d_values = np.random.uniform(0.0007, 0.003, (n_fits, 1))
    start_values = np.full(
        (n_fits, 1), ivim_mono_params.boundaries.start_values, dtype=np.float32
    )
    constraints = np.tile(
        np.concatenate(
            (
                ivim_mono_params.boundaries.lower_stop_values,
                ivim_mono_params.boundaries.upper_stop_values,
            ),
            dtype=np.float32,
        ),
        (n_fits, 1),
    )
    constraint_types = np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (1, 1))
    b_values = np.tile(ivim_mono_params.b_values.T, (n_fits, 1))

    for n_fit in range(n_fits):
        fit_args[n_fit, :] = np.squeeze(
            np.exp(-ivim_mono_params.b_values * d_values[n_fit])
        )

    result = gpufit.fit_constrained(
        fit_args,
        None,
        gpufit.ModelID.MONO_EXP_RED,
        initial_parameters=start_values,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=1e-3,
        max_number_iterations=250,
        parameters_to_fit=np.array([1], np.int32),
        estimator_id=gpufit.EstimatorID.LSE,
        user_info=b_values,
    )
    assert result is not None


@pytest.mark.gpu
def test_gpu_fit_tri(gpufit, decay_tri, ivim_tri_params):
    fit_data = decay_tri["fit_array"]
    n_fits = fit_data.shape[0]

    starts = [210, 0.001, 210, 0.02, 210, 0.01]
    lower = [10, 0.0007, 10, 0.003, 10, 0.003]
    upper = [2500, 0.05, 2500, 0.05, 2500, 0.3]
    start_values = np.tile(np.float32(starts), (n_fits, 1))
    constraints = np.tile(
        np.float32(list(zip(lower, upper))).flatten(),
        (n_fits, 1),
    )
    b_values = np.tile(np.char.array(ivim_tri_params.b_values.T), (n_fits, 1))
    constraint_types = np.squeeze(
        np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (6, 1))
    )

    result = gpufit.fit_constrained(
        fit_data,
        None,
        13,
        initial_parameters=start_values,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=0.000001,
        max_number_iterations=250,
        parameters_to_fit=None,
        estimator_id=gpufit.EstimatorID.LSE,
        user_info=b_values,
    )
    assert (result[0][0, :] != np.array(starts)).all()
