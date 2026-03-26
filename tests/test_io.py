"""Unit tests for I/O operations (NIfTI and b-values)."""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil

from pyneapple.io.nifti import (
    load_dwi_nifti,
    extract_2d_slice,
    save_parameter_map,
    normalize_dwi,
    create_mask,
)
from pyneapple.io.bvalue import (
    load_bvalues,
    save_bvalues,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_4d_data():
    """Create sample 4D DWI data."""
    return np.random.rand(32, 32, 10, 8).astype(np.float32)


@pytest.fixture
def sample_3d_data():
    """Create sample 3D data."""
    return np.random.rand(32, 32, 10).astype(np.float32)


@pytest.fixture
def sample_2d_params():
    """Create sample 2D parameter maps."""
    return {
        "S0": np.random.rand(32, 32) * 1000,
        "D": np.random.rand(32, 32) * 0.003,
    }


@pytest.fixture
def sample_bvalues():
    """Create sample b-values."""
    return np.array([0, 50, 100, 200, 400, 600, 800, 1000])


@pytest.fixture
def sample_nifti_4d(temp_dir, sample_4d_data):
    """Create a temporary 4D NIfTI file."""
    affine = np.eye(4)
    img = nib.Nifti1Image(sample_4d_data, affine)  # type: ignore
    path = Path(temp_dir) / "test_4d.nii.gz"
    nib.save(img, str(path))  # type: ignore
    return path, img


@pytest.fixture
def sample_nifti_3d(temp_dir, sample_3d_data):
    """Create a temporary 3D NIfTI file."""
    affine = np.eye(4)
    img = nib.Nifti1Image(sample_3d_data, affine)  # type: ignore
    path = Path(temp_dir) / "test_3d.nii.gz"
    nib.save(img, str(path))  # type: ignore
    return path, img


# ============================================================================
# Tests for NIfTI I/O
# ============================================================================


def test_load_dwi_nifti_4d(sample_nifti_4d):
    """Test loading 4D NIfTI file."""
    path, original_img = sample_nifti_4d

    data, img = load_dwi_nifti(str(path))

    assert data.ndim == 4
    assert data.shape == (32, 32, 10, 8)
    assert np.allclose(data, original_img.get_fdata())
    assert np.allclose(img.affine, original_img.affine)  # type: ignore


def test_load_dwi_nifti_3d(sample_nifti_3d):
    """Test loading 3D NIfTI file (should add dimension)."""
    path, original_img = sample_nifti_3d

    data, img = load_dwi_nifti(str(path))

    assert data.ndim == 4
    assert data.shape == (32, 32, 10, 1)
    # Check that the added dimension is correct
    assert np.allclose(data[..., 0], original_img.get_fdata())


def test_load_dwi_nifti_file_not_found():
    """Test error handling for non-existent file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dwi_nifti("nonexistent_file.nii.gz")

    assert "not found" in str(exc_info.value).lower()


def test_load_dwi_nifti_invalid_file(temp_dir):
    """Test error handling for invalid NIfTI file."""
    # Create a non-NIfTI file
    invalid_path = Path(temp_dir) / "invalid.nii.gz"
    with open(invalid_path, "w") as f:
        f.write("This is not a NIfTI file")

    with pytest.raises(ValueError) as exc_info:
        load_dwi_nifti(str(invalid_path))

    assert "failed to load" in str(exc_info.value).lower()


def test_extract_2d_slice(sample_4d_data):
    """Test extracting 2D slice from 4D volume."""
    slice_idx = 5
    slice_3d = extract_2d_slice(sample_4d_data, slice_idx)

    assert slice_3d.ndim == 3
    assert slice_3d.shape == (32, 32, 8)
    assert np.allclose(slice_3d, sample_4d_data[:, :, slice_idx, :])


def test_extract_2d_slice_first(sample_4d_data):
    """Test extracting first slice."""
    slice_3d = extract_2d_slice(sample_4d_data, 0)

    assert slice_3d.shape == (32, 32, 8)
    assert np.allclose(slice_3d, sample_4d_data[:, :, 0, :])


def test_extract_2d_slice_last(sample_4d_data):
    """Test extracting last slice."""
    slice_3d = extract_2d_slice(sample_4d_data, 9)

    assert slice_3d.shape == (32, 32, 8)
    assert np.allclose(slice_3d, sample_4d_data[:, :, 9, :])


def test_extract_2d_slice_invalid_dimensions():
    """Test error handling for non-4D input."""
    data_3d = np.random.rand(32, 32, 10)

    with pytest.raises(ValueError) as exc_info:
        extract_2d_slice(data_3d, 5)

    assert "expected 4d" in str(exc_info.value).lower()


def test_extract_2d_slice_out_of_bounds(sample_4d_data):
    """Test error handling for out-of-bounds slice index."""
    with pytest.raises(ValueError) as exc_info:
        extract_2d_slice(sample_4d_data, 10)  # Only 10 slices (0-9)

    assert "out of bounds" in str(exc_info.value).lower()


def test_extract_2d_slice_negative_index(sample_4d_data):
    """Test error handling for negative slice index."""
    with pytest.raises(ValueError) as exc_info:
        extract_2d_slice(sample_4d_data, -1)

    assert "out of bounds" in str(exc_info.value).lower()


def test_save_parameter_map_single(temp_dir, sample_2d_params, sample_nifti_4d):
    """Test saving a single parameter map."""
    _, ref_img = sample_nifti_4d
    output_path = Path(temp_dir) / "S0_map.nii.gz"

    save_parameter_map(sample_2d_params, str(output_path), ref_img, param_name="S0")

    # Verify file was created
    assert output_path.exists()

    # Load and verify
    saved_img = nib.load(str(output_path))  # type: ignore
    saved_data = saved_img.get_fdata()  # type: ignore

    # Should be 3D (2D with added dimension)
    assert saved_data.ndim == 3
    assert saved_data.shape[2] == 1
    assert np.allclose(saved_data[..., 0], sample_2d_params["S0"])
    assert np.allclose(saved_img.affine, ref_img.affine)  # type: ignore


def test_save_parameter_map_all(temp_dir, sample_2d_params, sample_nifti_4d):
    """Test saving all parameter maps."""
    _, ref_img = sample_nifti_4d
    output_path = Path(temp_dir) / "all_params.nii.gz"

    save_parameter_map(sample_2d_params, str(output_path), ref_img, param_name=None)

    # Verify file was created
    assert output_path.exists()

    # Load and verify
    saved_img = nib.load(str(output_path))  # type: ignore
    saved_data = saved_img.get_fdata()  # type: ignore

    # Should be 3D with 2 volumes
    assert saved_data.ndim == 3
    assert saved_data.shape[2] == 2


def test_save_parameter_map_invalid_param_name(
    temp_dir, sample_2d_params, sample_nifti_4d
):
    """Test error handling for invalid parameter name."""
    _, ref_img = sample_nifti_4d
    output_path = Path(temp_dir) / "invalid.nii.gz"

    with pytest.raises(ValueError) as exc_info:
        save_parameter_map(
            sample_2d_params, str(output_path), ref_img, param_name="invalid"
        )

    assert "not found" in str(exc_info.value).lower()


def test_save_parameter_map_inconsistent_shapes(temp_dir, sample_nifti_4d):
    """Test error handling for inconsistent parameter shapes."""
    _, ref_img = sample_nifti_4d
    output_path = Path(temp_dir) / "inconsistent.nii.gz"

    params = {
        "S0": np.random.rand(32, 32),
        "D": np.random.rand(16, 16),  # Different shape!
    }

    with pytest.raises(ValueError) as exc_info:
        save_parameter_map(params, str(output_path), ref_img, param_name=None)

    assert "inconsistent" in str(exc_info.value).lower()


def test_normalize_dwi_with_b0_indices(sample_4d_data):
    """Test DWI normalization with specified b0 indices."""
    # Use first two volumes as b0
    b0_indices = np.array([0, 1])

    normalized = normalize_dwi(sample_4d_data[:, :, 0, :], b0_indices)

    assert normalized.shape == sample_4d_data[:, :, 0, :].shape
    # After normalization, b0 images should be close to 1
    assert np.allclose(np.mean(normalized[..., b0_indices], axis=-1), 1.0, rtol=0.1)


def test_normalize_dwi_default_b0():
    """Test DWI normalization with default b0 (first volume)."""
    # Create data where first volume is constant (b0)
    data = np.ones((32, 32, 8))
    data[..., 0] = 2.0  # b0 has value 2
    data[..., 1:] = 1.0  # Other volumes have value 1

    normalized = normalize_dwi(data)

    # After normalization by b0=2, other volumes should be 0.5
    assert np.allclose(normalized[..., 0], 1.0)  # b0 normalized to 1
    assert np.allclose(normalized[..., 1:], 0.5)  # Others normalized


def test_normalize_dwi_zero_handling():
    """Test that zero values in b0 don't cause division by zero."""
    data = np.ones((32, 32, 8))
    data[0, 0, 0] = 0.0  # Zero value in b0

    # Should not raise error
    normalized = normalize_dwi(data)

    # Zero in b0 should remain zero (or close due to safe division)
    assert np.isfinite(normalized).all()


def test_create_mask_basic():
    """Test basic mask creation."""
    # Create data with clear foreground and background
    data = np.zeros((32, 32, 8))
    data[8:24, 8:24, :] = 1.0  # Central square has signal

    mask = create_mask(data, threshold=0.1)

    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    # Central region should be in mask
    assert np.all(mask[12:20, 12:20] == 1)
    # Corners should be background
    assert mask[0, 0] == 0
    assert mask[-1, -1] == 0


def test_create_mask_threshold_sensitivity():
    """Test mask creation with different thresholds."""
    # Create data with gradient
    data = np.zeros((32, 32, 8))
    data[8:24, 8:24, :] = 1.0  # Central square has signal

    mask_low = create_mask(data, threshold=0.05)
    mask_high = create_mask(data, threshold=0.5)

    # Lower threshold should include more voxels
    assert np.sum(mask_low) >= np.sum(mask_high)


def test_create_mask_all_zeros():
    """Test mask creation with all-zero data."""
    data = np.zeros((32, 32, 8))

    mask = create_mask(data, threshold=0.1)

    # Should be all zeros (no voxels above threshold)
    assert np.sum(mask) == 0


# ============================================================================
# Tests for B-value I/O
# ============================================================================


def test_load_bvalues_column_format(temp_dir):
    """Test loading b-values in column format (one per line)."""
    bval_path = Path(temp_dir) / "bvalues_column.txt"

    # Create test file
    with open(bval_path, "w") as f:
        f.write("0\n50\n100\n200\n400\n600\n800\n1000\n")

    bvalues = load_bvalues(str(bval_path))

    assert len(bvalues) == 8
    assert np.array_equal(bvalues, [0, 50, 100, 200, 400, 600, 800, 1000])
    assert bvalues.dtype == np.float64


def test_load_bvalues_row_format(temp_dir):
    """Test loading b-values in row format (space-separated)."""
    bval_path = Path(temp_dir) / "bvalues_row.txt"

    # Create test file
    with open(bval_path, "w") as f:
        f.write("0 50 100 200 400 600 800 1000\n")

    bvalues = load_bvalues(str(bval_path))

    assert len(bvalues) == 8
    assert np.array_equal(bvalues, [0, 50, 100, 200, 400, 600, 800, 1000])


def test_load_bvalues_mixed_format(temp_dir):
    """Test loading b-values in mixed format."""
    bval_path = Path(temp_dir) / "bvalues_mixed.txt"

    # Create test file with mixed format
    with open(bval_path, "w") as f:
        f.write("0 50 100\n")
        f.write("200\n")
        f.write("400 600\n")
        f.write("800 1000\n")

    bvalues = load_bvalues(str(bval_path))

    assert len(bvalues) == 8
    assert np.array_equal(bvalues, [0, 50, 100, 200, 400, 600, 800, 1000])


def test_load_bvalues_with_comments(temp_dir):
    """Test loading b-values with comment lines."""
    bval_path = Path(temp_dir) / "bvalues_comments.txt"

    # Create test file with comments
    with open(bval_path, "w") as f:
        f.write("# B-values for DWI acquisition\n")
        f.write("0\n")
        f.write("# Low b-values\n")
        f.write("50 100\n")
        f.write("# High b-values\n")
        f.write("200 400 600 800 1000\n")

    bvalues = load_bvalues(str(bval_path))

    assert len(bvalues) == 8
    assert np.array_equal(bvalues, [0, 50, 100, 200, 400, 600, 800, 1000])


def test_load_bvalues_with_empty_lines(temp_dir):
    """Test loading b-values with empty lines."""
    bval_path = Path(temp_dir) / "bvalues_empty.txt"

    # Create test file with empty lines
    with open(bval_path, "w") as f:
        f.write("0\n")
        f.write("\n")
        f.write("50 100\n")
        f.write("\n\n")
        f.write("200\n")

    bvalues = load_bvalues(str(bval_path))

    assert len(bvalues) == 4
    assert np.array_equal(bvalues, [0, 50, 100, 200])


def test_load_bvalues_floats(temp_dir):
    """Test loading b-values with floating-point values."""
    bval_path = Path(temp_dir) / "bvalues_floats.txt"

    # Create test file with floats
    with open(bval_path, "w") as f:
        f.write("0.0\n50.5\n100.2\n200.8\n")

    bvalues = load_bvalues(str(bval_path))

    assert len(bvalues) == 4
    assert np.allclose(bvalues, [0.0, 50.5, 100.2, 200.8])


def test_load_bvalues_file_not_found():
    """Test error handling for non-existent file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_bvalues("nonexistent_bvalues.txt")

    assert "not found" in str(exc_info.value).lower()


def test_load_bvalues_empty_file(temp_dir):
    """Test error handling for empty file."""
    bval_path = Path(temp_dir) / "bvalues_empty.txt"

    # Create empty file
    with open(bval_path, "w") as f:
        f.write("# Only comments\n")
        f.write("# No actual values\n")

    with pytest.raises(ValueError) as exc_info:
        load_bvalues(str(bval_path))

    assert "no b-values found" in str(exc_info.value).lower()


def test_load_bvalues_invalid_values(temp_dir):
    """Test error handling for invalid (non-numeric) values."""
    bval_path = Path(temp_dir) / "bvalues_invalid.txt"

    # Create file with invalid values
    with open(bval_path, "w") as f:
        f.write("0\n50\ninvalid\n200\n")

    with pytest.raises(ValueError) as exc_info:
        load_bvalues(str(bval_path))

    assert "invalid b-value" in str(exc_info.value).lower()


def test_load_bvalues_negative_values(temp_dir):
    """Test error handling for negative b-values."""
    bval_path = Path(temp_dir) / "bvalues_negative.txt"

    # Create file with negative values
    with open(bval_path, "w") as f:
        f.write("0\n50\n-100\n200\n")

    with pytest.raises(ValueError) as exc_info:
        load_bvalues(str(bval_path))

    assert "negative" in str(exc_info.value).lower()


def test_save_bvalues_column_format(temp_dir, sample_bvalues):
    """Test saving b-values in column format."""
    bval_path = Path(temp_dir) / "bvalues_out_column.txt"

    save_bvalues(sample_bvalues, str(bval_path), format="column")

    # Verify file was created
    assert bval_path.exists()

    # Load and verify
    loaded_bvalues = load_bvalues(str(bval_path))
    assert np.allclose(loaded_bvalues, sample_bvalues)


def test_save_bvalues_row_format(temp_dir, sample_bvalues):
    """Test saving b-values in row format."""
    bval_path = Path(temp_dir) / "bvalues_out_row.txt"

    save_bvalues(sample_bvalues, str(bval_path), format="row")

    # Verify file was created
    assert bval_path.exists()

    # Load and verify
    loaded_bvalues = load_bvalues(str(bval_path))
    assert np.allclose(loaded_bvalues, sample_bvalues)


def test_save_bvalues_invalid_dimensions(temp_dir):
    """Test error handling for non-1D b-values."""
    bval_path = Path(temp_dir) / "bvalues_out.txt"
    bvalues_2d = np.array([[0, 50], [100, 200]])

    with pytest.raises(ValueError) as exc_info:
        save_bvalues(bvalues_2d, str(bval_path))

    assert "expected 1d" in str(exc_info.value).lower()


def test_save_bvalues_negative_values(temp_dir):
    """Test error handling for negative b-values."""
    bval_path = Path(temp_dir) / "bvalues_out.txt"
    bvalues_neg = np.array([0, 50, -100, 200])

    with pytest.raises(ValueError) as exc_info:
        save_bvalues(bvalues_neg, str(bval_path))

    assert "negative" in str(exc_info.value).lower()


def test_save_bvalues_creates_directory(temp_dir):
    """Test that save_bvalues creates output directory if needed."""
    bval_path = Path(temp_dir) / "subdir" / "bvalues_out.txt"
    bvalues = np.array([0, 50, 100])

    # Directory doesn't exist yet
    assert not bval_path.parent.exists()

    save_bvalues(bvalues, str(bval_path))

    # Directory should now exist
    assert bval_path.parent.exists()
    assert bval_path.exists()


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_io_pipeline(temp_dir, sample_4d_data, sample_bvalues):
    """Test complete I/O pipeline: save and load."""
    # Save NIfTI
    affine = np.eye(4)
    img = nib.Nifti1Image(sample_4d_data, affine)  # type: ignore
    nifti_path = Path(temp_dir) / "test.nii.gz"
    nib.save(img, str(nifti_path))  # type: ignore

    # Save b-values
    bval_path = Path(temp_dir) / "bvalues.txt"
    save_bvalues(sample_bvalues, str(bval_path))

    # Load everything
    data, loaded_img = load_dwi_nifti(str(nifti_path))
    bvalues = load_bvalues(str(bval_path))

    # Extract slice
    slice_3d = extract_2d_slice(data, slice_idx=5)

    # Verify shapes match
    assert data.shape[:2] == slice_3d.shape[:2]
    assert data.shape[-1] == len(bvalues)
    assert slice_3d.shape[-1] == len(bvalues)


# ============================================================================
# Tests for TOML config — fixed_params
# ============================================================================


class TestTomlFixedParams:
    """Tests for [Fitting.model.fixed_params] TOML parsing."""

    def test_load_config_with_fixed_params(self, temp_dir):
        """fixed_params parsed from TOML produce correct model param_names."""
        from pyneapple.io.toml import load_config

        toml_text = """\
[Fitting]
fitter = "pixelwise"

[Fitting.model]
type = "monoexp"
fit_t1 = true
repetition_time = 3000.0

[Fitting.model.fixed_params]
T1 = 1200.0

[Fitting.solver]
type = "curvefit"
max_iter = 250
tol = 1e-8

[Fitting.solver.p0]
S0 = 900.0
D = 0.001

[Fitting.solver.bounds]
S0 = [1.0, 5000.0]
D = [0.0001, 0.1]
"""
        path = Path(temp_dir) / "fixed_params.toml"
        path.write_text(toml_text)

        config = load_config(path)
        assert config.fixed_params == {"T1": 1200.0}

        fitter = config.build_fitter()
        assert fitter.solver.model.param_names == ["S0", "D"]
        assert fitter.solver.model.fixed_params == {"T1": 1200.0}

    def test_load_config_no_fixed_params(self, temp_dir):
        """Config without fixed_params section works normally."""
        from pyneapple.io.toml import load_config

        toml_text = """\
[Fitting]
fitter = "pixelwise"

[Fitting.model]
type = "monoexp"

[Fitting.solver]
type = "curvefit"
max_iter = 250
tol = 1e-8

[Fitting.solver.p0]
S0 = 900.0
D = 0.001

[Fitting.solver.bounds]
S0 = [1.0, 5000.0]
D = [0.0001, 0.1]
"""
        path = Path(temp_dir) / "no_fixed.toml"
        path.write_text(toml_text)

        config = load_config(path)
        assert config.fixed_params == {}

        fitter = config.build_fitter()
        assert fitter.solver.model.param_names == ["S0", "D"]
