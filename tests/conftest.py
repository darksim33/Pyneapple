from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import signal

from pyneapple import (
    FitData,
    IVIMResults,
    NNLSCVParams,
    NNLSParams,
    NNLSResults,
)
from pyneapple.utils.logger import set_log_level
from radimgarray import RadImgArray, SegImgArray
from tests._files import *

# Import new test utilities
from tests.test_utils.signal_generators import IVIMSignalGenerator
from tests.test_utils.noise_models import SNRNoiseModel
from tests.test_utils import canonical_parameters as cp


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


# deploy_temp_file function removed - duplicate of ParameterTools.deploy_temp_file in _files.py


def get_spectrum(
    D_values: list,
    f_values: list,
    number_points: int,
    diffusion_range: tuple[float, float],
):
    """Calculate the diffusion spectrum for IVIM.

    The diffusion values have to be moved to take diffusion_range and number of
    points into account. The calculated spectrum is store inside the object.

    Args:
        number_points (int): Number of points in the diffusion spectrum.
        diffusion_range (tuple): Range of the diffusion
    """
    bins = np.array(
        np.logspace(
            np.log10(diffusion_range[0]),
            np.log10(diffusion_range[1]),
            number_points,
        )
    )
    spectrum = np.zeros(number_points)
    for d_value, fraction in zip(D_values, f_values):
        # Diffusion values are moved on range to calculate the spectrum
        index = np.unravel_index(
            np.argmin(abs(bins - d_value), axis=None),
            bins.shape,
        )[0].astype(int)

        spectrum += fraction * signal.unit_impulse(number_points, index)
    return spectrum


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
def results_biexp_pixel(b_values, signal_generator):
    """Generate bi-exponential fitting results for 8x8x2 pixel grid using kidney parameters.
    
    Uses kidney-specific parameter ranges (blood + tissue compartments).
    """
    b_values = np.array(b_values)
    shape = (8, 8, 2)
    n_bins = np.random.randint(250, 500)
    
    # Use global kidney parameter ranges
    f_range = cp.BLOOD_FRACTION_RANGE
    d1_range = cp.BLOOD_DIFFUSION_RANGE
    d2_range = cp.TISSUE_COMBINED_RANGE
    s0_range = cp.S0_RANGE
    
    f, D, S0, curve, spectrum, raw = {}, {}, {}, {}, {}, {}
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # Generate kidney-specific parameters
                f1 = np.random.uniform(*f_range)
                d1 = np.random.uniform(*d1_range)
                d2 = np.random.uniform(*d2_range)
                s0 = np.random.uniform(*s0_range)
                
                # Calculate signal using signal generator
                _curve = signal_generator.generate_biexp(b_values, f1=f1, D1=d1, D2=d2, S0=s0)
                
                # Store results
                f.update({(i, j, k): [f1, 1-f1]})  # Fractions sum to 1
                D.update({(i, j, k): [d1, d2]})
                S0.update({(i, j, k): s0})
                raw.update({(i, j, k): _curve})
                curve.update({(i, j, k): _curve})
                
                # Calculate spectrum
                _spectrum = get_spectrum(
                    [d1, d2],
                    [f1, 1-f1],
                    n_bins,
                    (min(d2_range), max(d1_range)),
                )
                spectrum.update({(i, j, k): _spectrum})
                
    return {"f": f, "D": D, "S0": S0, "curve": curve, "raw": raw, "spectrum": spectrum}


@pytest.fixture
def biexp_results_segmentation(b_values, signal_generator) -> dict[str, Any]:
    """Generate bi-exponential fitting results for segmentation labels using kidney parameters.
    
    Uses kidney-specific parameter ranges (blood + tissue compartments).
    """
    b_values = np.array(b_values)
    n_segs = np.random.randint(1, 10)
    n_bins = np.random.randint(250, 500)
    
    # Use global kidney parameter ranges
    f_range = cp.BLOOD_FRACTION_RANGE
    d1_range = cp.BLOOD_DIFFUSION_RANGE
    d2_range = cp.TISSUE_COMBINED_RANGE
    s0_range = cp.S0_RANGE
    
    f, D, S0, curve, spectrum, raw = {}, {}, {}, {}, {}, {}
    
    for seg in range(n_segs):
        # Generate kidney-specific parameters
        f1 = np.random.uniform(*f_range)
        d1 = np.random.uniform(*d1_range)
        d2 = np.random.uniform(*d2_range)
        s0 = np.random.uniform(*s0_range)
        
        # Calculate signal using signal generator
        _curve = signal_generator.generate_biexp(b_values, f1=f1, D1=d1, D2=d2, S0=s0)
        
        # Store results
        f.update({seg: [f1, 1-f1]})  # Fractions sum to 1
        D.update({seg: [d1, d2]})
        S0.update({seg: s0})
        raw.update({seg: _curve})
        curve.update({seg: _curve})
        
        # Calculate spectrum
        _spectrum = get_spectrum(
            [d1, d2],
            [f1, 1-f1],
            n_bins,
            (min(d2_range), max(d1_range)),
        )
        spectrum.update({seg: _spectrum})

    return {"f": f, "D": D, "S0": S0, "curve": curve, "raw": raw, "spectrum": spectrum}


@pytest.fixture
def results_with_t1(results_biexp_pixel):
    """Add T1 values to biexp results."""
    results = IVIMResults(IVIMParams())
    results.load_from_dict(results_biexp_pixel)
    t1 = {}
    for key in results_biexp_pixel["D"].keys():
        t1.update({key: np.random.uniform(800, 1500)})
    results.t1.update(t1)
    results.params.fit_model.fit_t1 = True
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


# ============================================================================
# New SNR-based synthetic signal fixtures
# ============================================================================


@pytest.fixture(scope="module")
def signal_generator():
    """Module-scoped IVIM signal generator instance."""
    return IVIMSignalGenerator()


@pytest.fixture(scope="module")
def noise_model():
    """Module-scoped SNR noise model instance."""
    return SNRNoiseModel()


# Canonical parameter fixtures (module-scoped for reuse)


@pytest.fixture(scope="module")
def canonical_b_values():
    """Standard b-values for IVIM testing (16 values for better fitting)."""
    return cp.STANDARD_B_VALUES


@pytest.fixture(scope="module")
def canonical_mono_params():
    """Canonical mono-exponential parameters."""
    return cp.MONO_TYPICAL


@pytest.fixture(scope="module")
def canonical_biexp_params():
    """Canonical bi-exponential parameters (typical perfusion)."""
    return cp.BIEXP_TYPICAL


@pytest.fixture(scope="module")
def canonical_biexp_low_perf():
    """Canonical bi-exponential parameters (low perfusion)."""
    return cp.BIEXP_LOW_PERFUSION


@pytest.fixture(scope="module")
def canonical_biexp_high_perf():
    """Canonical bi-exponential parameters (high perfusion)."""
    return cp.BIEXP_HIGH_PERFUSION


@pytest.fixture(scope="module")
def canonical_triexp_params():
    """Canonical tri-exponential parameters."""
    return cp.TRIEXP_TYPICAL


# Clean signal fixtures (function-scoped, no noise)


@pytest.fixture
def clean_mono_signal(signal_generator, canonical_b_values, canonical_mono_params):
    """Generate clean mono-exponential signal without noise."""
    return signal_generator.generate_monoexp(
        canonical_b_values,
        D=canonical_mono_params["D"],
        S0=canonical_mono_params["S0"],
    )


@pytest.fixture
def clean_biexp_signal(signal_generator, canonical_b_values, canonical_biexp_params):
    """Generate clean bi-exponential signal without noise."""
    return signal_generator.generate_biexp(
        canonical_b_values,
        f1=canonical_biexp_params["f1"],
        D1=canonical_biexp_params["D1"],
        D2=canonical_biexp_params["D2"],
        S0=canonical_biexp_params["S0"],
    )


@pytest.fixture
def clean_triexp_signal(signal_generator, canonical_b_values, canonical_triexp_params):
    """Generate clean tri-exponential signal without noise."""
    return signal_generator.generate_triexp(
        canonical_b_values,
        f1=canonical_triexp_params["f1"],
        D1=canonical_triexp_params["D1"],
        f2=canonical_triexp_params["f2"],
        D2=canonical_triexp_params["D2"],
        D3=canonical_triexp_params["D3"],
        S0=canonical_triexp_params["S0"],
    )


# Noisy signal fixtures (function-scoped, default SNR=140)


@pytest.fixture
def noisy_mono_signal(clean_mono_signal, noise_model):
    """Generate mono-exponential signal with SNR=140 noise (kidney quality)."""
    return noise_model.add_noise(clean_mono_signal, snr=cp.DEFAULT_SNR, seed=cp.DEFAULT_SEED)


@pytest.fixture
def noisy_biexp_signal(clean_biexp_signal, noise_model):
    """Generate bi-exponential signal with SNR=140 noise (kidney quality)."""
    return noise_model.add_noise(clean_biexp_signal, snr=cp.DEFAULT_SNR, seed=cp.DEFAULT_SEED)


@pytest.fixture
def noisy_triexp_signal(clean_triexp_signal, noise_model):
    """Generate tri-exponential signal with SNR=140 noise (kidney quality)."""
    return noise_model.add_noise(clean_triexp_signal, snr=cp.DEFAULT_SNR, seed=cp.DEFAULT_SEED)


# Multi-pixel array fixtures for GPU testing (function-scoped)


@pytest.fixture
def decay_mono_array(signal_generator, noise_model, canonical_b_values):
    """Generate 8x8x2 array of mono-exponential signals with SNR=140 noise (kidney quality).
    
    Compatible replacement for old decay_mono fixture.
    Returns dict with fit_array for GPU batch fitting.
    """
    shape = (8, 8, 2)
    n_pixels = np.prod(shape)
    
    # Use global kidney parameter ranges
    d_values = np.random.uniform(*cp.TISSUE_COMBINED_RANGE, n_pixels)
    s0_values = np.random.uniform(*cp.S0_RANGE, n_pixels)
    
    # Generate clean signals for all pixels
    signals = np.array([
        signal_generator.generate_monoexp(canonical_b_values, D=d, S0=s0)
        for d, s0 in zip(d_values, s0_values)
    ], dtype=np.float32)
    
    # Add noise to all signals
    noisy_signals = np.array([
        noise_model.add_noise(sig, snr=cp.DEFAULT_SNR, seed=cp.DEFAULT_SEED + i)
        for i, sig in enumerate(signals)
    ], dtype=np.float32)
    
    indexes = list(np.ndindex(shape))
    fit_args = zip(indexes, noisy_signals)
    
    return {
        "fit_args": fit_args,
        "fit_array": noisy_signals,
        "d_values": d_values.reshape(n_pixels, 1, 1),
        "s0_values": s0_values.reshape(n_pixels, 1, 1),
    }


@pytest.fixture
def decay_bi_array(signal_generator, noise_model, canonical_b_values):
    """Generate 8x8x2 array of bi-exponential signals with SNR=140 noise (kidney quality).
    
    Compatible replacement for old decay_bi fixture.
    Uses kidney-specific parameter ranges (blood + tissue compartments).
    """
    shape = (8, 8, 2)
    n_pixels = np.prod(shape)
    
    # Use global kidney parameter ranges
    f1_values = np.random.uniform(*cp.BLOOD_FRACTION_RANGE, n_pixels)
    d1_values = np.random.uniform(*cp.BLOOD_DIFFUSION_RANGE, n_pixels)
    d2_values = np.random.uniform(*cp.TISSUE_COMBINED_RANGE, n_pixels)
    s0_values = np.random.uniform(*cp.S0_RANGE, n_pixels)
    
    # Generate clean signals
    signals = np.array([
        signal_generator.generate_biexp(canonical_b_values, f1=f1, D1=d1, D2=d2, S0=s0)
        for f1, d1, d2, s0 in zip(f1_values, d1_values, d2_values, s0_values)
    ], dtype=np.float32)
    
    # Add noise
    noisy_signals = np.array([
        noise_model.add_noise(sig, snr=cp.DEFAULT_SNR, seed=cp.DEFAULT_SEED + i)
        for i, sig in enumerate(signals)
    ], dtype=np.float32)
    
    indexes = list(np.ndindex(shape))
    fit_args = zip(indexes, noisy_signals)
    
    return {
        "fit_args": fit_args,
        "fit_array": noisy_signals,
        "f1_values": f1_values.reshape(n_pixels, 1, 1),
        "d1_values": d1_values.reshape(n_pixels, 1, 1),
        "d2_values": d2_values.reshape(n_pixels, 1, 1),
        "s0_values": s0_values.reshape(n_pixels, 1, 1),
    }


@pytest.fixture
def decay_tri_array(signal_generator, noise_model, canonical_b_values):
    """Generate 8x8x2 array of tri-exponential signals with SNR=140 noise (kidney quality).
    
    Compatible replacement for old decay_tri fixture.
    Uses kidney-specific parameter ranges (blood + tubule + tissue compartments).
    """
    shape = (8, 8, 2)
    n_pixels = np.prod(shape)
    
    # Use global kidney parameter ranges
    f1_values = np.random.uniform(*cp.BLOOD_FRACTION_RANGE, n_pixels)
    d1_values = np.random.uniform(*cp.BLOOD_DIFFUSION_RANGE, n_pixels)
    f2_values = np.random.uniform(*cp.TUBULE_FRACTION_RANGE, n_pixels)
    d2_values = np.random.uniform(*cp.TUBULE_DIFFUSION_RANGE, n_pixels)
    d3_values = np.random.uniform(*cp.TISSUE_DIFFUSION_RANGE, n_pixels)
    s0_values = np.random.uniform(*cp.S0_RANGE, n_pixels)
    
    # Generate clean signals
    signals = np.array([
        signal_generator.generate_triexp(
            canonical_b_values, f1=f1, D1=d1, f2=f2, D2=d2, D3=d3, S0=s0
        )
        for f1, d1, f2, d2, d3, s0 in zip(
            f1_values, d1_values, f2_values, d2_values, d3_values, s0_values
        )
    ], dtype=np.float32)
    
    # Add noise
    noisy_signals = np.array([
        noise_model.add_noise(sig, snr=cp.DEFAULT_SNR, seed=cp.DEFAULT_SEED + i)
        for i, sig in enumerate(signals)
    ], dtype=np.float32)
    
    indexes = list(np.ndindex(shape))
    fit_args = zip(indexes, noisy_signals)
    
    return {
        "fit_args": fit_args,
        "fit_array": noisy_signals,
        "f1_values": f1_values.reshape(n_pixels, 1, 1),
        "d1_values": d1_values.reshape(n_pixels, 1, 1),
        "f2_values": f2_values.reshape(n_pixels, 1, 1),
        "d2_values": d2_values.reshape(n_pixels, 1, 1),
        "d3_values": d3_values.reshape(n_pixels, 1, 1),
        "s0_values": s0_values.reshape(n_pixels, 1, 1),
    }


# Parametrized SNR fixture for custom noise levels


@pytest.fixture
def custom_snr_signal(request, signal_generator, noise_model, canonical_b_values):
    """Generate signal with custom SNR level.
    
    Usage in test:
        @pytest.mark.parametrize("custom_snr_signal", [(params, snr)], indirect=True)
        def test_something(custom_snr_signal):
            signal, params, snr = custom_snr_signal
    """
    params, snr = request.param
    
    # Determine signal type and generate clean signal
    if "f2" in params and "D3" in params:
        # Tri-exponential (has f2 and D3)
        clean_signal = signal_generator.generate_triexp(
            canonical_b_values,
            f1=params["f1"],
            D1=params["D1"],
            f2=params["f2"],
            D2=params["D2"],
            D3=params["D3"],
            S0=params["S0"],
        )
    elif "f1" in params:
        # Bi-exponential (has f1 but not D3)
        clean_signal = signal_generator.generate_biexp(
            canonical_b_values,
            f1=params["f1"],
            D1=params["D1"],
            D2=params["D2"],
            S0=params["S0"],
        )
    else:
        # Mono-exponential
        clean_signal = signal_generator.generate_monoexp(
            canonical_b_values,
            D=params["D"],
            S0=params["S0"],
        )
    
    # Add noise
    noisy_signal = noise_model.add_noise(clean_signal, snr=snr, seed=cp.DEFAULT_SEED)
    
    return noisy_signal, params, snr
