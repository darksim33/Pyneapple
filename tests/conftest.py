from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from pyneapple import (
    FitData,
    NNLSCVParams,
    NNLSParams,
    NNLSResults,
)
from pyneapple.parameters.parameters import BaseParams
from pyneapple.results.results import BaseResults
from pyneapple.utils.logger import set_log_level
from radimgarray import RadImgArray, SegImgArray
from tests._files import *
from tests._parameters import *


def pytest_configure(config):
    # Perform setup tasks here
    # Check if requirements are met
    requirements_met()


def requirements_met():
    # Check if requirements are met

    root = Path(__file__).parent.parent
    print(root)

    # Check dir
    if not (root / "tests/.data").is_dir():
        raise RuntimeError(
            "Requirements not met. No '.data' directory. Tests cannot proceed."
        )

    if not (root / "tests/.temp").is_dir():
        (root / "tests/.temp").mkdir(exist_ok=True)

    # Check files
    if not (root / r"tests/.data/test_img.nii.gz").is_file():
        raise RuntimeError(
            "Requirements not met. No 'test_img.nii' file. Tests cannot proceed."
        )
    if not (root / r"tests/.data/test_seg.nii.gz").is_file():
        raise RuntimeError(
            "Requirements not met. No 'test_seg.nii' file. Tests cannot proceed."
        )
    if not (root / r"tests/.data/test_bvalues.bval").is_file():
        raise RuntimeError(
            "Requirements not met. No 'b_values' file. Tests cannot proceed."
        )

    return True


@pytest.fixture(scope="session", autouse=True)
def setup_logger():
    """Setup logger for pytest - lightweight version."""
    # Set log level to ERROR for tests (minimal output)
    set_log_level("ERROR")
    yield


def pytest_collection_modifyitems(config, items):
    # Run Tests in specific order
    sorted_items = items.copy()

    file_mapping = {item: item.location[0] for item in items}
    file_order = ["model", "parameters", "fitting", "results"]
    for file in file_order:
        sorted_items = [it for it in sorted_items if file not in file_mapping[it]] + [
            it for it in sorted_items if file in file_mapping[it]
        ]

    model_order = ["ivim", "nnls", "ideal"]
    for model in model_order:
        sorted_items = [it for it in sorted_items if model not in file_mapping[it]] + [
            it for it in sorted_items if model in file_mapping[it]
        ]
    items[:] = sorted_items


def deploy_temp_file(file: Path | str):
    """Yield file and unlink afterwards."""
    if isinstance(file, str):
        file = Path(file)
    yield file
    if file.exists():
        file.unlink()


# Fixtures for testing


@pytest.fixture
def root():
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir(root):
    """Temporary directory for tests."""
    temp_path = root / "tests/.temp"
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


@pytest.fixture
def img(root):
    file = root / "tests" / ".data" / "test_img.nii.gz"
    yield RadImgArray(file)


@pytest.fixture
def seg(root):
    file = root / "tests" / ".data" / "test_seg_48p.nii.gz"
    seg = SegImgArray(file)
    seg = seg[:, :, :, np.newaxis] if seg.ndim == 3 else seg  # Ensure it has 4 dims
    yield seg


@pytest.fixture
def seg_reduced():
    array = np.ones((2, 2, 2, 1))
    nii = SegImgArray(array)
    return nii


# --- FitData ---


@pytest.fixture
def ivim_mono_fit_data(img, seg, ivim_mono_params_file):
    fit_data = FitData(
        img,
        seg,
        ivim_mono_params_file,
    )
    return fit_data


@pytest.fixture
def ivim_bi_fit_data(img, seg, ivim_bi_params_file):
    fit_data = FitData(
        img,
        seg,
        ivim_bi_params_file,
    )
    return fit_data


@pytest.fixture
def ivim_tri_fit_data(img, seg, ivim_tri_params_file):
    fit_data = FitData(
        img,
        seg,
        ivim_tri_params_file,
    )
    return fit_data


@pytest.fixture
def results_bi_exp(seg: SegImgArray):
    shape = np.squeeze(seg).shape
    d_slow_map = np.random.rand(*shape)
    d_fast_map = np.random.rand(*shape)
    f_slow_map = np.random.randint(1, 2500, shape)
    f_fast_map = np.random.randint(1, 2500, shape)

    results = []
    for idx in np.squeeze(seg).get_seg_indices(1):
        results.append(
            (
                idx,
                np.array(
                    [f_slow_map[idx], d_slow_map[idx], f_slow_map[idx], d_fast_map[idx]]
                ),
            )
        )

    return results


@pytest.fixture
def fixed_values(seg: SegImgArray):  # Segmented Fitting related
    shape = np.squeeze(seg).shape
    d_slow_map = np.zeros(shape)
    d_slow_map[np.squeeze(seg) > 0] = np.random.rand() * 10**-5
    t1_map = np.zeros(shape)
    t1_map[np.squeeze(seg) > 0] = np.random.randint(1, 2500)
    d_slow, t1 = {}, {}
    # result = []
    for idx in list(zip(*np.where(np.squeeze(seg) > 0))):
        d_slow[idx] = d_slow_map[idx]
        t1[idx] = t1_map[idx]

    return d_slow, t1
    # result.append((idx, np.array([d_slow_map[idx], t1_map[idx]])))
    # return result


@pytest.fixture
def b_values():
    return [
        0,
        10,
        20,
        30,
        40,
        50,
        70,
        100,
        150,
        200,
        250,
        350,
        450,
        550,
        650,
        750,
    ]


# --- NNLS ---


@pytest.fixture
def nnls_params(nnls_params_file):
    return NNLSParams(nnls_params_file)


@pytest.fixture
def nnlscv_params(nnls_cv_params_file):
    return NNLSCVParams(nnls_cv_params_file)


@pytest.fixture
def nnls_fit_data(img, seg, nnls_params_file):
    fit_data = FitData(
        img,
        seg,
        nnls_params_file,
    )
    fit_data.params.max_iter = 10000
    return fit_data


@pytest.fixture
def nnls_fit_results(nnls_params) -> tuple:
    def random_pixel() -> tuple:
        pixel = list()
        value_range = [0, 10]
        for i in range(3):
            pixel.append(random.randint(value_range[0], value_range[1]))
        return tuple(pixel)

    def get_random_pixel_pos(number_components: int) -> list:
        pixels = list()
        while len(pixels) < number_components:
            point = random_pixel()
            if point not in pixels:
                pixels.append(point)
        return pixels

    n_components = np.random.randint(2, 10)
    pixel_pos = get_random_pixel_pos(n_components)
    spectra_list = list()
    d_values_dict = dict()
    f_values_dict = dict()
    for n in range(n_components):
        # Get D Values from bins
        bins = nnls_params.get_bins()
        d_value_indexes = random.sample(
            np.linspace(0, len(bins) - 1, num=len(bins)).astype(int).tolist(), 3
        )
        d_values = np.array([bins[i] for i in d_value_indexes])

        # Get f Values
        f1 = random.uniform(0, 1)
        f2 = random.uniform(0, 1)
        while f1 + f2 >= 1:
            f1 = random.uniform(0, 1)
            f2 = random.uniform(0, 1)
        f3 = 1 - f1 - f2
        f_values = np.array([f1, f2, f3])

        # Get Spectrum
        spectrum = np.zeros(nnls_params.boundaries["n_bins"])
        for idx, d in enumerate(d_value_indexes):
            spectrum = spectrum + f_values[idx] * signal.unit_impulse(
                nnls_params.boundaries["n_bins"],
                d_value_indexes[idx],
            )

        spectra_list.append((pixel_pos[n], spectrum))
        d_values_dict[pixel_pos[n]] = d_values
        f_values_dict[pixel_pos[n]] = f_values

    return spectra_list, d_values_dict, f_values_dict, pixel_pos


@pytest.fixture
def nnls_fit_results_data(nnls_fit_results, nnls_params):
    result = NNLSResults(nnls_params)
    fit_results = result.eval_results(nnls_fit_results[0])
    result.update_results(fit_results)
    return result


@pytest.fixture
def nnlscv_fit_data(img, seg, nnlscv_params_file):
    fit_data = FitData(
        img,
        seg,
        nnlscv_params_file,
    )
    fit_data.params.max_iter = 10000
    return fit_data


@pytest.fixture
def decay_mono(ivim_mono_params) -> dict:
    shape = (8, 8, 2)
    b_values = ivim_mono_params.b_values[np.newaxis, :, :]
    indexes = list(np.ndindex(shape))
    d_values = np.random.uniform(0.0007, 0.003, (int(np.prod(shape)), 1, 1))
    f_values = np.random.randint(150, 250, (int(np.prod(shape)), 1, 1))
    decay = np.sum(f_values * np.exp(-b_values * d_values), axis=2, dtype=np.float32)
    fit_args = zip(
        (indexes[i] for i in range(len(indexes))),
        (decay[i, :] for i in range(len(indexes))),
    )
    return {
        "fit_args": fit_args,
        "fit_array": decay,
        "d_values": d_values,
        "f_values": f_values,
    }


@pytest.fixture
def decay_bi(ivim_bi_params):
    shape = (8, 8, 2)
    b_values = ivim_bi_params.b_values[np.newaxis, :, :]
    indexes = list(np.ndindex(shape))
    d_slow = np.random.uniform(0.0007, 0.003, (int(np.prod(shape)), 1, 1))
    d_fast = np.random.uniform(0.01, 0.3, (int(np.prod(shape)), 1, 1))
    f_values = np.random.randint(150, 250, (int(np.prod(shape)), 1, 1))
    d_values = np.concatenate((d_slow, d_fast), axis=2)
    decay = np.sum(f_values * np.exp(-b_values * d_values), axis=2, dtype=np.float32)
    fit_args = zip(
        (indexes[i] for i in range(len(indexes))),
        (decay[i, :] for i in range(len(indexes))),
    )
    return {
        "fit_args": fit_args,
        "fit_array": decay,
        "d_values": d_values,
        "f_values": f_values,
    }


@pytest.fixture
def decay_tri(ivim_tri_params) -> dict:
    shape = (8, 8, 2)
    b_values = ivim_tri_params.b_values[np.newaxis, :, :]
    indexes = list(np.ndindex(shape))
    # d_values = np.random.uniform(0.0007, 0.003, (int(np.prod(shape)), 1, 3))
    d_slow = np.random.uniform(0.0007, 0.003, (int(np.prod(shape)), 1, 1))
    d_iter = np.random.uniform(0.003, 0.01, (int(np.prod(shape)), 1, 1))
    d_fast = np.random.uniform(0.01, 0.3, (int(np.prod(shape)), 1, 1))
    d_values = np.concatenate((d_slow, d_iter, d_fast), axis=2)

    f_values = np.random.randint(100, 300, (int(np.prod(shape)), 1, 3))
    decay = np.sum(f_values * np.exp(-b_values * d_values), axis=2, dtype=np.float32)
    fit_args = zip(
        (indexes[i] for i in range(len(indexes))),
        (decay[i, :] for i in range(len(indexes))),
    )
    return {
        "fit_args": fit_args,
        "fit_array": decay,
        "d_values": d_values,
        "f_values": f_values,
    }
