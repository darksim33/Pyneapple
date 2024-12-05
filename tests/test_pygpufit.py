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
def test_gpu_fit_mono(gpufit, decay_mono, ivim_mono_params):
    fit_data = decay_mono["fit_array"]
    n_fits = fit_data.shape[0]

    starts = [210, 0.0015]
    lower = [10, 0.0007]
    upper = [2500, 0.003]
    start_values = np.tile(np.float32(starts), (n_fits, 1))
    constraints = np.tile(
        np.float32(list(zip(lower, upper))).flatten(),
        (n_fits, 1),
    )

    constraint_types = np.squeeze(
        np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (2, 1))
    )
    b_values = np.squeeze(ivim_mono_params.b_values).astype(np.float32)

    result = gpufit.fit_constrained(
        fit_data,
        None,
        gpufit.ModelID.MONO_EXP,
        initial_parameters=start_values,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=1e-7,
        max_number_iterations=250,
        parameters_to_fit=None,
        estimator_id=gpufit.EstimatorID.LSE,
        user_info=b_values,
    )
    assert np.mean(result[3]) > 3


@pytest.mark.gpu
def test_gpu_fit_tri(gpufit, decay_tri, ivim_tri_params):
    fit_data = decay_tri["fit_array"]
    n_fits = fit_data.shape[0]

    starts = [210, 0.001, 210, 0.02, 210, 0.01]
    lower = [10, 0.0007, 10, 0.003, 10, 0.05]
    upper = [2500, 0.003, 2500, 0.01, 2500, 0.3]
    start_values = np.tile(np.float32(starts), (n_fits, 1))
    constraints = np.tile(
        np.float32(list(zip(lower, upper))).flatten(),
        (n_fits, 1),
    )

    b_values = np.squeeze(ivim_tri_params.b_values).astype(np.float32)

    constraint_types = np.squeeze(
        np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (6, 1))
    )

    result = gpufit.fit_constrained(
        fit_data,
        None,
        gpufit.ModelID.TRI_EXP,
        initial_parameters=start_values,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=0.00001,
        max_number_iterations=250,
        parameters_to_fit=None,
        estimator_id=gpufit.EstimatorID.LSE,
        user_info=b_values,
    )
    assert np.mean(result[3]) > 15


def test_gpu_fitter(decay_tri, ivim_tri_gpu_params):
    fit_args = decay_tri["fit_args"]
    results = gpu_fitter(fit_args, ivim_tri_gpu_params)
    assert results is not None
