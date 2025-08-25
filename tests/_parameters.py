import pytest
from pyneapple import (
    IVIMParams,
    NNLSParams,
    NNLSCVParams,
    NNLSResults,
    IVIMSegmentedParams,
)

# --- Parameters ---


@pytest.fixture
def ivim_mono_params(ivim_mono_params_file):
    yield IVIMParams(ivim_mono_params_file)


# --- Bi
# Exponential Fitting ---


@pytest.fixture
def ivim_bi_params(ivim_bi_params_file):
    yield IVIMParams(ivim_bi_params_file)


@pytest.fixture
def ivim_bi_segmented_params(ivim_bi_segmented_params_file):
    return IVIMSegmentedParams(ivim_bi_segmented_params_file)


@pytest.fixture
def ivim_bi_gpu_params(ivim_bi_params_file):
    params = IVIMParams(ivim_bi_params_file)
    params.fit_type = "GPU"
    return params


# --- Tri Exponential Fitting ---


@pytest.fixture
def ivim_tri_params(ivim_tri_params_file):
    return IVIMParams(ivim_tri_params_file)


@pytest.fixture
def ivim_tri_t1_segmented_params(ivim_tri_t1_segmented_params_file):
    return IVIMSegmentedParams(ivim_tri_t1_segmented_params_file)


@pytest.fixture
def ivim_tri_gpu_params(ivim_tri_params_file):
    params = IVIMParams(ivim_tri_params_file)
    params.fit_type = "GPU"
    return params
