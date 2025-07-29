from __future__ import annotations
import pytest
import sys
import random
import json
import copy
import tempfile
import functools
import numpy as np
from scipy import signal
from pathlib import Path

from pyneapple.utils.logger import logger, set_log_level

# from pyneapple.fit import parameters, FitData, Results
from pyneapple import (
    IVIMParams,
    NNLSParams,
    NNLSCVParams,
    NNLSResults,
    IVIMSegmentedParams,
)

from pyneapple import FitData

from pyneapple.results.results import BaseResults
from radimgarray import RadImgArray, SegImgArray


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


def create_modified_ivim_params_json(
    base_file_path: Path, output_dir: Path = None, **modifications
):
    """
    Loads an IVIM parameter JSON file, modifies specific settings and
    saves the result as a temporary file.

    Args:
        base_file_path: Path to the base JSON file
        output_dir: Optional output path (if None, a temporary directory is used)
        **modifications: Key-value pairs of parameters to modify

    Yields:
        Path to the generated temporary JSON file
    """
    # Load base JSON file
    with open(base_file_path, "r") as f:
        params = json.load(f)

    # Create deep copy to avoid modifying the original
    modified_params = copy.deepcopy(params)

    for key, value in modifications.items():
        keys = key.split("__")
        modified_params[keys[0]][keys[1]] = value
        if key == "Model__fit_t1":
            modified_params["boundaries"].update({"T": {"1": [2000, 10, 10000]}})

    # Prepare output file
    if output_dir is None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
    else:
        temp_dir = output_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        base_name = base_file_path.name
        file_name = f"modified_{base_name}"
        temp_path = temp_dir / file_name

    try:
        # Save modified parameters to JSON file
        with open(temp_path, "w") as f:
            json.dump(modified_params, f, indent=2)
        return temp_path
    finally:
        #     # Clean up temporary file
        #     if temp_path.exists():
        #         temp_path.unlink()
        pass


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
def img(root):
    file = root / r"tests/.data/test_img.nii.gz"
    if file.exists():
        assert True
    else:
        assert False
    return RadImgArray(file)


@pytest.fixture
def seg(root):
    file = root / r"tests/.data/test_seg_48p.nii.gz"
    if file.exists():
        assert True
    else:
        assert file.exists()
    img = SegImgArray(file)
    if img.ndim == 3:
        img = img[:, :, :, np.newaxis]
    return img


@pytest.fixture
def seg_reduced():
    array = np.ones((2, 2, 2, 1))
    nii = SegImgArray(array)
    return nii


@pytest.fixture
def out_json(root):
    return root / r"tests/.temp/test_params.json"


@pytest.fixture
def out_nii(root):
    return root / r"tests/.temp/out_nii.nii.gz"


@pytest.fixture
def out_excel(root):
    file = root / r"tests/.temp/out_excel.xlsx"
    yield file
    if file.is_file():
        file.unlink()


# IVIM
@pytest.fixture
def ivim_mono_params_file(root):
    return root / r"tests/.data/fitting/params_monoexp.json"


@pytest.fixture
def ivim_mono_params(ivim_mono_params_file):
    if ivim_mono_params_file.exists():
        assert True
    else:
        assert False
    return IVIMParams(ivim_mono_params_file)


@pytest.fixture
def ivim_bi_params_file(root):
    return root / r"tests/.data/fitting/params_biexp.json"


@pytest.fixture
def ivim_bi_t1_params_file(ivim_bi_params_file):
    yield from deploy_temp_file(
        create_modified_ivim_params_json(
            ivim_bi_params_file, Model__fit_t1=True, Model__mixing_time=20
        )
    )


@pytest.fixture
def ivim_bi_params(ivim_bi_params_file):
    if ivim_bi_params_file.exists():
        assert True
    else:
        assert False
    return IVIMParams(ivim_bi_params_file)


@pytest.fixture
def ivim_bi_segmented_params_file(root):
    return root / r"tests/.data/fitting/params_biexp_segmented.json"


@pytest.fixture
def ivim_bi_segmented_params(ivim_bi_segmented_params_file):
    if ivim_bi_segmented_params_file.exists():
        assert True
    else:
        assert False
    return IVIMSegmentedParams(ivim_bi_segmented_params_file)


@pytest.fixture
def ivim_bi_gpu_params_file(ivim_bi_params_file):
    yield from deploy_temp_file(
        create_modified_ivim_params_json(ivim_bi_params_file, General__fit_type="GPU")
    )


@pytest.fixture
def ivim_bi_gpu_params(ivim_bi_gpu_params_file):
    return IVIMParams(ivim_bi_gpu_params_file)


@pytest.fixture
def ideal_params_file():
    """Create a temporary IDEAL parameter file."""
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
        f.write(
            b"""
            # Test IDEAL Parameter File
            [General]
            Class = "IDEALParams"
            fit_type = "single"
            max_iter = 100
            fit_tolerance = 1e-6
            n_pools = 4
            ideal_dims = 2
            step_tol = [0.01, 0.02, 0.03, 0.04]
            dim_steps = [[1, 1], [2, 2], [4, 4], [8, 8], [16, 16], [32, 32]]
            seg_threshold = 0.025

            [Model]
            model = "BiExp"
            fit_reduced = false
            fit_S0 = true

            [boundaries]

            [boundaries.D]
            "1" = [0.001, 0.0007, 0.05]
            "2" = [0.02, 0.003, 0.3]

            [boundaries.f]
            "1" = [85, 10, 500]
            "2" = [20, 1, 100]

            """
        )
        temp_file = f.name

    yield Path(temp_file)

    # Clean up
    if Path(temp_file).exists():
        Path(temp_file).unlink()


# Tri Exponential Fitting


@pytest.fixture
def ivim_tri_params_file(root):
    return root / r"tests/.data/fitting/params_triexp.json"


@pytest.fixture
def ivim_tri_t1_params_file(ivim_tri_params_file):
    yield from deploy_temp_file(
        create_modified_ivim_params_json(
            ivim_tri_params_file, Model__fit_t1=True, Model__mixing_time=20
        )
    )


@pytest.fixture
def ivim_tri_t1_no_mixing_params_file(ivim_tri_params_file):
    yield from deploy_temp_file(
        create_modified_ivim_params_json(ivim_tri_params_file, Model__fit_t1=True)
    )


@pytest.fixture
def ivim_tri_params(ivim_tri_params_file):
    if ivim_tri_params_file.exists():
        assert True
    else:
        assert False
    return IVIMParams(ivim_tri_params_file)


@pytest.fixture
def ivim_tri_segmented_params_file(root):
    return root / r"tests/.data/fitting/params_triexp_segmented.json"


@pytest.fixture
def ivim_tri_t1_segmented_params_file(ivim_tri_segmented_params_file):
    yield create_modified_ivim_params_json(
        ivim_tri_segmented_params_file, Model__fit_t1=True, Model__mixing_time=20
    )


@pytest.fixture
def ivim_tri_t1_segmented_params(ivim_tri_t1_segmented_params_file):
    if not ivim_tri_t1_segmented_params_file.is_file():
        assert False
    return IVIMSegmentedParams(ivim_tri_t1_segmented_params_file)


@pytest.fixture
def ivim_tri_gpu_params_file(ivim_tri_params_file):
    yield from deploy_temp_file(
        create_modified_ivim_params_json(ivim_tri_params_file, General__fit_type="GPU")
    )


@pytest.fixture
def ivim_tri_gpu_params(ivim_tri_gpu_params_file):
    if not ivim_tri_gpu_params_file.is_file():
        assert False
    return IVIMParams(ivim_tri_gpu_params_file)


# FitData


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


# NNLS
@pytest.fixture
def nnls_params_file(root):
    file = root / r"tests/.data/fitting/params_nnls.json"
    if file.exists():
        assert True
    else:
        assert False
    return file


@pytest.fixture
def nnlscv_params_file(root):
    file = root / r"tests/.data/fitting/params_nnls_cv.json"
    if file.exists():
        assert True
    else:
        assert False
    return file


@pytest.fixture
def nnls_params(nnls_params_file):
    if nnls_params_file.exists():
        assert True
    else:
        assert False
    return NNLSParams(nnls_params_file)


@pytest.fixture
def nnlscv_params(nnlscv_params_file):
    if nnlscv_params_file.exists():
        assert True
    else:
        assert False
    return NNLSCVParams(nnlscv_params_file)


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
        spectrum = np.zeros(nnls_params.boundaries.number_points)
        for idx, d in enumerate(d_value_indexes):
            spectrum = spectrum + f_values[idx] * signal.unit_impulse(
                nnls_params.boundaries.number_points,
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
def random_results(ivim_tri_params):
    f = {(0, 0, 0): [1.1, 1.2, 1.3]}
    d = {(0, 0, 0): [1.0, 1.2, 1.3]}
    s_0 = {(0, 0, 0): np.random.rand(1)}
    results = BaseResults(ivim_tri_params)
    results.f.update(f)
    results.D.update(d)
    results.S0.update(s_0)
    return results


@pytest.fixture
def array_result():
    """Random decay signal."""
    spectrum = np.zeros((2, 2, 1, 11))
    bins = np.linspace(0, 10, 11)
    for index in np.ndindex((2, 2)):
        spectrum[index] = np.exp(-np.kron(bins, abs(np.random.randn(1))))
    return spectrum


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
