"""Tests for io export functions: reconstruct_maps, save_spectrum_to_nifti,
save_params_to_excel, save_spectrum_to_excel, save_params_to_hdf5."""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

from pyneapple.io.nifti import reconstruct_maps, save_spectrum_to_nifti
from pyneapple.io.hdf5 import save_params_to_hdf5, load_from_hdf5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_params():
    """Fitted params dict: 4 pixels, one scalar parameter each."""
    return {"D": np.array([0.001, 0.002, 0.003, 0.004], dtype=np.float32)}


@pytest.fixture
def multi_param():
    """Fitted params with two scalar parameters."""
    return {
        "D": np.array([0.001, 0.002, 0.003, 0.004], dtype=np.float32),
        "f": np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32),
    }


@pytest.fixture
def pixel_indices():
    """Four pixel coordinates in a (4, 4, 1) volume."""
    return [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)]


@pytest.fixture
def spatial_shape():
    """Spatial shape matching the pixel_indices fixture."""
    return (4, 4, 1)


@pytest.fixture
def ref_nifti():
    """Minimal reference NIfTI image."""
    return nib.Nifti1Image(np.zeros((4, 4, 1)), affine=np.eye(4))


@pytest.fixture
def spectrum_params(pixel_indices):
    """Spectrum array: 4 pixels × 10 bins."""
    rng = np.random.default_rng(0)
    arr = np.abs(rng.standard_normal((4, 10))).astype(np.float32)
    return arr


@pytest.fixture
def bins():
    """Log-spaced bin array with 10 entries."""
    return np.logspace(-4, -1, 10)


# ---------------------------------------------------------------------------
# reconstruct_maps
# ---------------------------------------------------------------------------


class TestReconstructMaps:
    """Tests for reconstruct_maps()."""

    def test_output_keys_match_input(self, simple_params, pixel_indices, spatial_shape):
        """Output dict has the same keys as fitted_params."""
        maps = reconstruct_maps(simple_params, pixel_indices, spatial_shape)
        assert set(maps.keys()) == set(simple_params.keys())

    def test_output_shape_matches_spatial_shape(
        self, simple_params, pixel_indices, spatial_shape
    ):
        """Each output array has the given spatial shape."""
        maps = reconstruct_maps(simple_params, pixel_indices, spatial_shape)
        assert maps["D"].shape == spatial_shape

    def test_value_placed_at_correct_position(
        self, simple_params, pixel_indices, spatial_shape
    ):
        """Values appear at the expected spatial index."""
        maps = reconstruct_maps(simple_params, pixel_indices, spatial_shape)
        for i, idx in enumerate(pixel_indices):
            np.testing.assert_allclose(maps["D"][idx], simple_params["D"][i])

    def test_unvisited_voxels_are_zero(
        self, simple_params, pixel_indices, spatial_shape
    ):
        """Positions not in pixel_indices are zero-filled."""
        maps = reconstruct_maps(simple_params, pixel_indices, spatial_shape)
        assert maps["D"][0, 1, 0] == pytest.approx(0.0)

    def test_dtype_is_float32(self, simple_params, pixel_indices, spatial_shape):
        """Output arrays are float32."""
        maps = reconstruct_maps(simple_params, pixel_indices, spatial_shape)
        assert maps["D"].dtype == np.float32

    def test_multi_dim_values_produce_4d_volume(self, pixel_indices, spatial_shape):
        """Values with shape (n_pixels, k) produce a (X, Y, Z, k) volume."""
        params = {"spec": np.ones((4, 5), dtype=np.float32)}
        maps = reconstruct_maps(params, pixel_indices, spatial_shape)
        assert maps["spec"].shape == spatial_shape + (5,)

    def test_multiple_params_returned(
        self, multi_param, pixel_indices, spatial_shape
    ):
        """All parameters are included in the output."""
        maps = reconstruct_maps(multi_param, pixel_indices, spatial_shape)
        assert "D" in maps and "f" in maps


# ---------------------------------------------------------------------------
# save_spectrum_to_nifti
# ---------------------------------------------------------------------------


class TestSaveSpectrumToNifti:
    """Tests for save_spectrum_to_nifti()."""

    def test_creates_file(
        self, spectrum_params, pixel_indices, spatial_shape, tmp_path, ref_nifti
    ):
        """Output NIfTI file is created on disk."""
        out = tmp_path / "spec.nii.gz"
        save_spectrum_to_nifti(
            spectrum_params, pixel_indices, spatial_shape, out, ref_nifti
        )
        assert out.exists()

    def test_output_is_4d(
        self, spectrum_params, pixel_indices, spatial_shape, bins, tmp_path, ref_nifti
    ):
        """Saved volume has shape (X, Y, Z, n_bins)."""
        out = tmp_path / "spec.nii.gz"
        save_spectrum_to_nifti(
            spectrum_params, pixel_indices, spatial_shape, out, ref_nifti
        )
        loaded = nib.load(str(out)).get_fdata()  # type: ignore
        assert loaded.shape == spatial_shape + (spectrum_params.shape[1],)

    def test_values_placed_correctly(
        self, spectrum_params, pixel_indices, spatial_shape, tmp_path
    ):
        """Spectrum values appear at the correct spatial location."""
        out = tmp_path / "spec.nii.gz"
        save_spectrum_to_nifti(spectrum_params, pixel_indices, spatial_shape, out)
        loaded = nib.load(str(out)).get_fdata().astype(np.float32)  # type: ignore
        np.testing.assert_allclose(
            loaded[pixel_indices[0]], spectrum_params[0], rtol=1e-5
        )

    def test_raises_for_non_2d_spectrum(
        self, pixel_indices, spatial_shape, tmp_path
    ):
        """ValueError raised when spectrum is not 2-D."""
        with pytest.raises(ValueError, match="2-D"):
            save_spectrum_to_nifti(
                np.ones((4,)), pixel_indices, spatial_shape, tmp_path / "x.nii"
            )

    def test_no_reference_nifti_uses_identity_affine(
        self, spectrum_params, pixel_indices, spatial_shape, tmp_path
    ):
        """Saving without a reference nifti uses an identity affine."""
        out = tmp_path / "spec_no_ref.nii.gz"
        save_spectrum_to_nifti(spectrum_params, pixel_indices, spatial_shape, out)
        img = nib.load(str(out))
        np.testing.assert_allclose(img.affine, np.eye(4))    # type: ignore


# ---------------------------------------------------------------------------
# save_params_to_excel
# ---------------------------------------------------------------------------


class TestSaveParamsToExcel:
    """Tests for save_params_to_excel()."""

    def test_creates_file(self, multi_param, pixel_indices, tmp_path):
        """Excel file is created on disk."""
        pytest.importorskip("pandas")
        from pyneapple.io.excel import save_params_to_excel

        out = tmp_path / "params.xlsx"
        save_params_to_excel(multi_param, pixel_indices, out)
        assert out.exists()

    def test_row_count_matches_pixels(self, multi_param, pixel_indices, tmp_path):
        """One row per pixel (excluding header)."""
        pd = pytest.importorskip("pandas")
        from pyneapple.io.excel import save_params_to_excel

        out = tmp_path / "params.xlsx"
        save_params_to_excel(multi_param, pixel_indices, out)
        df = pd.read_excel(out)
        assert len(df) == len(pixel_indices)

    def test_coordinates_in_columns(self, multi_param, pixel_indices, tmp_path):
        """Output DataFrame has 'x' and 'y' columns."""
        pd = pytest.importorskip("pandas")
        from pyneapple.io.excel import save_params_to_excel

        out = tmp_path / "params.xlsx"
        save_params_to_excel(multi_param, pixel_indices, out)
        df = pd.read_excel(out)
        assert "x" in df.columns and "y" in df.columns

    def test_raises_without_pandas(self, multi_param, pixel_indices, tmp_path, mocker):
        """ImportError raised when pandas is not available."""
        from pyneapple.io import excel

        mocker.patch.dict("sys.modules", {"pandas": None})
        with pytest.raises(ImportError, match="pandas"):
            excel.save_params_to_excel(multi_param, pixel_indices, tmp_path / "x.xlsx")

    def test_raises_on_empty_params(self, pixel_indices, tmp_path):
        """ValueError raised when fitted_params is empty."""
        pytest.importorskip("pandas")
        from pyneapple.io.excel import save_params_to_excel

        with pytest.raises(ValueError):
            save_params_to_excel({}, pixel_indices, tmp_path / "x.xlsx")


# ---------------------------------------------------------------------------
# save_spectrum_to_excel
# ---------------------------------------------------------------------------


class TestSaveSpectrumToExcel:
    """Tests for save_spectrum_to_excel()."""

    def test_creates_file(
        self, spectrum_params, pixel_indices, bins, tmp_path
    ):
        """Excel file is created on disk."""
        pytest.importorskip("pandas")
        from pyneapple.io.excel import save_spectrum_to_excel

        out = tmp_path / "spectrum.xlsx"
        save_spectrum_to_excel(spectrum_params, pixel_indices, bins, out)
        assert out.exists()

    def test_column_count_includes_bins(
        self, spectrum_params, pixel_indices, bins, tmp_path
    ):
        """Output DataFrame has coordinate columns + n_bins columns."""
        pd = pytest.importorskip("pandas")
        from pyneapple.io.excel import save_spectrum_to_excel

        out = tmp_path / "spectrum.xlsx"
        save_spectrum_to_excel(spectrum_params, pixel_indices, bins, out)
        df = pd.read_excel(out)
        # x, y, z, + n_bins columns
        assert df.shape[1] >= spectrum_params.shape[1]

    def test_raises_for_non_2d_spectrum(self, pixel_indices, bins, tmp_path):
        """ValueError raised when spectrum is not 2-D."""
        pytest.importorskip("pandas")
        from pyneapple.io.excel import save_spectrum_to_excel

        with pytest.raises(ValueError, match="2-D"):
            save_spectrum_to_excel(
                np.ones((4,)), pixel_indices, bins, tmp_path / "x.xlsx"
            )

    def test_raises_on_bin_mismatch(
        self, spectrum_params, pixel_indices, tmp_path
    ):
        """ValueError raised when bin count doesn't match spectrum columns."""
        pytest.importorskip("pandas")
        from pyneapple.io.excel import save_spectrum_to_excel

        wrong_bins = np.linspace(0, 1, 5)  # wrong length
        with pytest.raises(ValueError, match="bins"):
            save_spectrum_to_excel(
                spectrum_params, pixel_indices, wrong_bins, tmp_path / "x.xlsx"
            )


# ---------------------------------------------------------------------------
# save_params_to_hdf5
# ---------------------------------------------------------------------------


class TestSaveParamsToHdf5:
    """Tests for save_params_to_hdf5()."""

    def test_creates_file(
        self, multi_param, pixel_indices, spatial_shape, tmp_path
    ):
        """HDF5 file is created on disk."""
        out = tmp_path / "results.h5"
        save_params_to_hdf5(multi_param, pixel_indices, spatial_shape, out)
        assert out.exists()

    def test_loaded_params_match(
        self, multi_param, pixel_indices, spatial_shape, tmp_path
    ):
        """Loaded parameter values match the reconstructed maps."""
        from pyneapple.io.nifti import reconstruct_maps

        out = tmp_path / "results.h5"
        save_params_to_hdf5(multi_param, pixel_indices, spatial_shape, out)
        loaded = load_from_hdf5(out)
        expected = reconstruct_maps(multi_param, pixel_indices, spatial_shape)
        np.testing.assert_allclose(
            loaded["params"]["D"], expected["D"], rtol=1e-5
        )

    def test_raises_on_empty_params(
        self, pixel_indices, spatial_shape, tmp_path
    ):
        """ValueError raised when fitted_params is empty."""
        with pytest.raises(ValueError):
            save_params_to_hdf5({}, pixel_indices, spatial_shape, tmp_path / "x.h5")
