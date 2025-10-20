"""Test Basic Parameter Class and Boundaries"""

import pytest
import numpy as np
from pyneapple.parameters.parameters import BaseParams
from radimgarray import RadImgArray, SegImgArray


def test_load_b_values(root):
    parameters = BaseParams()
    file = root / r"tests/.data/test_bvalues.bval"
    assert file.is_file()
    parameters.load_b_values(file)
    b_values = np.array(
        [
            0,
            50,
            100,
            150,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            1000,
            1200,
            1400,
            1600,
            1800,
        ]
    )
    assert b_values.all() == parameters.b_values.all()


def test_get_pixel_args(img, seg):
    parameters = BaseParams()
    args = parameters.get_pixel_args(img, seg)
    assert len(list(args)) == len(np.where(seg != 0)[0])


@pytest.mark.parametrize("seg_number", [1, 2])
def test_get_seg_args_seg_number(img, seg, seg_number):
    parameters = BaseParams()
    args = parameters.get_seg_args(img, seg, seg_number)
    assert len(list(args)) == 1


# --- Get Segment Arguments ---


@pytest.mark.parametrize(
    "seed", [42, 123, 456]
)  # Multiple random seeds for reproducibility
def test_get_seg_args_mean(seed):
    """Test that get_seg_args correctly calculates mean signal for segments with random data"""

    np.random.seed(seed)

    # Random test dimensions
    n_slices = np.random.randint(2, 6)
    height = np.random.randint(3, 8)
    width = np.random.randint(3, 8)
    n_volumes = np.random.randint(50, 150)

    # Random base signal parameters
    amplitude = np.random.uniform(10, 50)
    decay_rate = np.random.uniform(0.0005, 0.002)
    x_max = np.random.uniform(800, 1200)

    # Generate base signal
    x_data = np.linspace(0, x_max, n_volumes)
    base_signal = amplitude * np.exp(-x_data * decay_rate)

    # Create image and segmentation arrays
    img = np.zeros((n_slices, height, width, n_volumes))
    seg = np.zeros((n_slices, height, width, 1), dtype=int)

    # Random segment number
    seg_number = np.random.randint(1, 5)

    # Random noise levels for each voxel
    noise_levels = np.random.uniform(-5, 5, (n_slices, height, width))

    # Track which voxels belong to the segment and calculate expected mean
    segment_signals = []
    voxel_count = 0

    for i in range(n_slices):
        for j in range(height):
            for k in range(width):
                # Randomly assign some voxels to the segment
                if np.random.random() > 0.3:  # 70% chance of being in segment
                    seg[i, j, k, :] = seg_number
                    voxel_signal = base_signal + noise_levels[i, j, k]
                    img[i, j, k, :] = voxel_signal
                    segment_signals.append(voxel_signal)
                    voxel_count += 1

    # Skip test if no voxels in segment
    if voxel_count == 0:
        pytest.skip("No voxels assigned to segment")

    # Calculate expected mean manually
    expected_mean = np.mean(segment_signals, axis=0)

    # Test the function
    img = RadImgArray(img)
    seg = SegImgArray(seg)
    parameters = BaseParams()
    args = list(parameters.get_seg_args(img, seg, seg_number))

    assert len(args) == 1, f"Expected 1 argument tuple, got {len(args)}"
    _, mean_signal = args[0]

    # Assert the calculated mean matches expected
    np.testing.assert_allclose(mean_signal, expected_mean, rtol=1e-10)

    # Additional assertions
    assert (
        len(mean_signal) == n_volumes
    ), f"Mean signal length {len(mean_signal)} != {n_volumes}"
    assert not np.any(np.isnan(mean_signal)), "Mean signal contains NaN values"
    assert not np.any(np.isinf(mean_signal)), "Mean signal contains infinite values"


def test_get_seg_args_mean_multiple_segments():
    """Test mean calculation with multiple different segments"""

    np.random.seed(789)

    # Fixed dimensions for clarity
    img = np.zeros((2, 3, 3, 50))
    seg = np.zeros((2, 3, 3, 1), dtype=int)

    # Create different base signals for different segments
    x_data = np.linspace(0, 1000, 50)
    signal_1 = 10 * np.exp(-x_data * 0.001)
    signal_2 = 20 * np.exp(-x_data * 0.002)

    # Assign voxels to different segments with known signals
    seg[0, 0, 0, :] = 1
    seg[0, 0, 1, :] = 1
    seg[1, 1, 1, :] = 2
    seg[1, 2, 2, :] = 2

    img[0, 0, 0, :] = signal_1 + 1
    img[0, 0, 1, :] = signal_1 - 1
    img[1, 1, 1, :] = signal_2 + 2
    img[1, 2, 2, :] = signal_2 - 2

    expected_mean_1 = signal_1  # (signal_1 + 1 + signal_1 - 1) / 2
    expected_mean_2 = signal_2  # (signal_2 + 2 + signal_2 - 2) / 2

    img = RadImgArray(img)
    seg = SegImgArray(seg)
    parameters = BaseParams()

    # Test segment 1
    args_1 = list(parameters.get_seg_args(img, seg, 1))
    _, mean_signal_1 = args_1[0]
    np.testing.assert_allclose(mean_signal_1, expected_mean_1, rtol=1e-10)

    # Test segment 2
    args_2 = list(parameters.get_seg_args(img, seg, 2))
    _, mean_signal_2 = args_2[0]
    np.testing.assert_allclose(mean_signal_2, expected_mean_2, rtol=1e-10)


def test_get_seg_args_mean_edge_cases():
    """Test edge cases for segment mean calculation"""

    # Test with single voxel segment
    img = np.random.rand(2, 2, 2, 30)
    seg = np.zeros((2, 2, 2, 1), dtype=int)
    seg[0, 0, 0, :] = 1  # Only one voxel in segment

    img_array = RadImgArray(img)
    seg_array = SegImgArray(seg)
    parameters = BaseParams()

    args = list(parameters.get_seg_args(img_array, seg_array, 1))
    _, mean_signal = args[0]

    # For single voxel, mean should equal the voxel signal
    np.testing.assert_allclose(mean_signal, img[0, 0, 0, :], rtol=1e-10)

    # Test with nonexistent segment
    with pytest.raises(ValueError):
        list(parameters.get_seg_args(img_array, seg_array, 999))


def test_boundaries_ivim(ivim_bi_params):
    start_values = np.random.randint(2, 100, 4)
    lower_bound = start_values - 1
    upper_bound = start_values + 1
    bounds = {
        "D": {
            "slow": np.array([start_values[1], lower_bound[1], upper_bound[1]]),
            "fast": np.array([start_values[3], lower_bound[3], upper_bound[3]]),
        },
        "f": {
            "slow": np.array([start_values[0], lower_bound[0], upper_bound[0]]),
            "fast": np.array([start_values[2], lower_bound[2], upper_bound[2]]),
        },
    }
    ivim_bi_params.boundaries.update(bounds)
    assert (start_values == ivim_bi_params.boundaries.start_values).all()
    assert (lower_bound == ivim_bi_params.boundaries.lower_bounds).all()
    assert (upper_bound == ivim_bi_params.boundaries.upper_bounds).all()

    start_values = np.random.randint(2, 100, 3)
    lower_bound = start_values - 1
    upper_bound = start_values + 1
    bounds = {
        "D": {
            "slow": np.array([start_values[1], lower_bound[1], upper_bound[1]]),
            "fast": np.array([start_values[2], lower_bound[2], upper_bound[2]]),
        },
        "f": {
            "slow": np.array([start_values[0], lower_bound[0], upper_bound[0]]),
        },
    }
    ivim_bi_params.boundaries.clear()
    ivim_bi_params.boundaries.update(bounds)
    assert (start_values == ivim_bi_params.boundaries.start_values).all()
    assert (lower_bound == ivim_bi_params.boundaries.lower_bounds).all()
    assert (upper_bound == ivim_bi_params.boundaries.upper_bounds).all()
