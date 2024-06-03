import pytest
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox
import random
import numpy as np
from scipy import signal

from pyneapple.fit import parameters, FitData, Results
from pyneapple.utils.nifti import Nii, NiiSeg


def pytest_configure(config):
    # Perform setup tasks here
    # Check if requirements are met
    requirements_met()


def requirements_met():
    # Check if requirements are met

    # Check dir
    if not Path(r".data").is_dir():
        raise RuntimeError(
            "Requirements not met. No '.data' directory. Tests cannot proceed."
        )

    if not Path(r".out").is_dir():
        Path(".out").mkdir(exist_ok=True)

    # Check files
    if not Path(r".data/test_img.nii.gz").is_file():
        raise RuntimeError(
            "Requirements not met. No 'test_img.nii' file. Tests cannot proceed."
        )
    if not Path(r".data/test_seg.nii.gz").is_file():
        raise RuntimeError(
            "Requirements not met. No 'test_seg.nii' file. Tests cannot proceed."
        )
    if not Path(r".data/test_bvalues.bval").is_file():
        raise RuntimeError(
            "Requirements not met. No 'b_values' file. Tests cannot proceed."
        )

    return True


@pytest.fixture
def img():
    file = Path(r".data/test_img.nii.gz")
    if file.exists():
        assert True
    else:
        assert False
    return Nii(file)


@pytest.fixture
def seg():
    file = Path(r".data/test_seg_48p.nii.gz")
    if file.exists():
        assert True
    else:
        assert file.exists()
    return NiiSeg(file)


@pytest.fixture
def seg_reduced():
    array = np.ones((2, 2, 2, 1))
    nii = NiiSeg().from_array(array)
    return nii


@pytest.fixture
def out_json():
    file = Path(r".out/test_params.json")
    yield file
    if file.is_file():
        file.unlink()


@pytest.fixture
def out_nii():
    file = Path(r".out/out_nii.nii.gz")
    yield file
    if file.is_file():
        file.unlink()


@pytest.fixture
def out_excel():
    file = Path(r".out/out_excel.xlsx")
    yield file
    if file.is_file():
        file.unlink()


# IVIM
@pytest.fixture
def ivim_mono_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_mono.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IVIMParams(file)


@pytest.fixture
def ivim_bi_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_bi.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IVIMParams(file)


@pytest.fixture
def ivim_tri_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IVIMParams(file)


@pytest.fixture
def ivim_mono_fit_data(img, seg, ivim_mono_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.params = ivim_mono_params
    return fit_data


@pytest.fixture
def ivim_bi_fit_data(img, seg, ivim_bi_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.params = ivim_bi_params
    return fit_data


@pytest.fixture
def ivim_tri_fit_data(img, seg, ivim_tri_params):
    fit_data = FitData(
        "IVIM",
        None,
        img,
        seg,
    )
    fit_data.params = ivim_tri_params
    return fit_data


# NNLS
@pytest.fixture
def nnls_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.NNLSParams(file)


@pytest.fixture
def nnlscv_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.NNLSCVParams(file)


@pytest.fixture
def nnls_fit_data(img, seg, nnls_params):
    fit_data = FitData(
        "NNLS",
        None,
        img,
        seg,
    )
    fit_data.params = nnls_params
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
    results = Results()
    fit_results = nnls_params.eval_fitting_results(nnls_fit_results[0])
    results.update_results(fit_results)
    return results


@pytest.fixture
def nnlscv_fit_data(img, seg, nnlscv_params):
    fit_data = FitData(
        "NNLSCV",
        None,
        img,
        seg,
    )
    fit_data.params = nnlscv_params
    fit_data.params.max_iter = 10000
    return fit_data


# IDEAL
@pytest.fixture
def ideal_params():
    file = Path(r"../src/pyneapple/resources/fitting/default_params_IDEAL_bi.json")
    if file.exists():
        assert True
    else:
        assert False
    return parameters.IDEALParams(file)


@pytest.fixture
def test_ideal_fit_data(img, seg, ideal_params):
    fit_data = FitData(
        "IDEAL",
        None,
        img,
        seg,
    )
    fit_data.params = ideal_params
    return fit_data


@pytest.fixture
def app():
    application = QApplication([])
    yield application
    application.quit()


@pytest.fixture
def message_box():
    message_box = QMessageBox()
    yield message_box
    message_box.close()
