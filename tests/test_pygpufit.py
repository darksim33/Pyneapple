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
        "BIEXP",
        "BIEXP_RED",
        "TRIEXP",
        "TRIEXP_RED",
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
        gpufit.ModelID.TRIEXP,
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

@pytest.mark.gpu
def test_gpu_fitter_with_general_boundaries(gpufit, decay_mono, ivim_mono_params):
    """Test gpu_fitter with general boundaries."""
    pixel_indices = [(0, 0), (0, 1), (1, 0)]
    data_list = [decay_mono["fit_array"][i] for i in range(len(pixel_indices))]
    zipped_data = zip(pixel_indices, data_list)
    
    result = gpu_fitter(zipped_data, ivim_mono_params)
    
    assert len(result) == len(pixel_indices)
    assert all(isinstance(r[0], tuple) for r in result)
    assert all(isinstance(r[1], np.ndarray) for r in result)


@pytest.mark.gpu
def test_gpu_fitter_with_individual_boundaries(gpufit, decay_tri, ivim_tri_params):
    """Test gpu_fitter with individual boundaries for each pixel."""
    from pyneapple.parameters.boundaries import IVIMBoundaryDict
    
    pixel_indices = [(0, 0), (0, 1), (1, 0)]
    data_list = [decay_tri["fit_array"][i] for i in range(len(pixel_indices))]
    zipped_data = zip(pixel_indices, data_list)
    
    # Create individual boundaries
    individual_bounds = IVIMBoundaryDict({
        "f": {
            "f1": {(0, 0): [210, 10, 2500], (0, 1): [200, 10, 2500], (1, 0): [220, 10, 2500]},
            "f2": {(0, 0): [210, 10, 2500], (0, 1): [200, 10, 2500], (1, 0): [220, 10, 2500]},
            "f3": {(0, 0): [210, 10, 2500], (0, 1): [200, 10, 2500], (1, 0): [220, 10, 2500]},
        },
        "D": {
            "D1": {(0, 0): [0.001, 0.0007, 0.003], (0, 1): [0.0012, 0.0007, 0.003], (1, 0): [0.0008, 0.0007, 0.003]},
            "D2": {(0, 0): [0.02, 0.003, 0.01], (0, 1): [0.015, 0.003, 0.01], (1, 0): [0.025, 0.003, 0.01]},
            "D3": {(0, 0): [0.01, 0.05, 0.3], (0, 1): [0.012, 0.05, 0.3], (1, 0): [0.008, 0.05, 0.3]},
        },
    })
    
    ivim_tri_params.boundaries = individual_bounds
    
    result = gpu_fitter(zipped_data, ivim_tri_params)
    
    assert len(result) == len(pixel_indices)
    for i, (pixel_idx, fit_result) in enumerate(result):
        assert pixel_idx == pixel_indices[i]
        assert isinstance(fit_result, np.ndarray)
        assert len(fit_result) == 6  # 6 parameters for triexp


@pytest.mark.gpu
def test_gpu_fitter_validates_boundaries_order(gpufit, decay_mono, ivim_mono_params):
    """Test that boundaries are applied in correct order."""
    from pyneapple.parameters.boundaries import IVIMBoundaryDict
    
    pixel_indices = [(0, 0)]
    data_list = [decay_mono["fit_array"][0]]
    zipped_data = zip(pixel_indices, data_list)
    
    # Set specific boundaries
    tight_bounds = IVIMBoundaryDict({
        "f": {"f1": [100, 90, 110]},
        "D": {"D1": [0.002, 0.0019, 0.0021]},
    })
    
    ivim_mono_params.boundaries = tight_bounds
    
    result = gpu_fitter(zipped_data, ivim_mono_params)
    
    assert len(result) == 1
    # Check that fitted values are within tight bounds
    assert 90 <= result[0][1][0] <= 110
    assert 0.0019 <= result[0][1][1] <= 0.0021


@pytest.mark.gpu
def test_gpu_fitter_raises_on_non_zipped_data(gpufit, decay_mono, ivim_mono_params):
    """Test that gpu_fitter raises error when data is not zipped."""
    non_zipped_data = [decay_mono["fit_array"][0]]
    
    with pytest.raises(ValueError, match="Data for GPU fitting must be zipped"):
        gpu_fitter(non_zipped_data, ivim_mono_params)


@pytest.mark.gpu
def test_gpu_fitter_raises_on_invalid_model(gpufit, decay_mono, ivim_mono_params):
    """Test that gpu_fitter raises error for invalid model."""
    pixel_indices = [(0, 0)]
    data_list = [decay_mono["fit_array"][0]]
    zipped_data = zip(pixel_indices, data_list)
    
    ivim_mono_params.model = "INVALID_MODEL"
    
    with pytest.raises(ValueError, match="Invalid model for GPU fitting"):
        gpu_fitter(zipped_data, ivim_mono_params)