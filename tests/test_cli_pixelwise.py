"""Tests for the pyneapple-pixelwise CLI tool.

Covers:
- _reconstruct_maps() spatial reconstruction helper
- _build_parser() argument definitions
- main() exit codes and output artefacts (integration tests using tmp_path)
"""

import textwrap
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pyneapple.models import MonoExpModel
from pyneapple.cli.pixelwise import _build_parser, _reconstruct_maps, main

# ---------------------------------------------------------------------------
# N_BINS used across NNLS helpers (small value for fast tests)
# ---------------------------------------------------------------------------
_NNLS_N_BINS = 10


# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=float)


def _write_bval(path: Path, bvalues: np.ndarray = B_VALUES) -> Path:
    """Write b-values one per line to *path*."""
    path.write_text("\n".join(str(int(b)) for b in bvalues))
    return path


def _write_nifti(path: Path, data: np.ndarray) -> Path:
    """Save *data* as a NIfTI file with identity affine."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))  # type: ignore
    return path


def _make_dwi(
    n_x: int = 4, n_y: int = 4, n_z: int = 1, S0: float = 1000.0, D: float = 0.001
) -> np.ndarray:
    """Synthetic noise-free monoexp DWI of shape (n_x, n_y, n_z, N_B)."""
    signal = MonoExpModel().forward(B_VALUES, S0, D)
    return np.tile(signal, (n_x, n_y, n_z, 1))


def _write_monoexp_config(path: Path) -> Path:
    """Write a minimal monoexp pixelwise TOML config to *path*."""
    path.write_text(
        textwrap.dedent(
            """\
        [Fitting]
        fitter = "pixelwise"

        [Fitting.model]
        type = "monoexp"

        [Fitting.solver]
        type = "curvefit"
        max_iter = 250
        tol = 1e-8

        [Fitting.solver.p0]
        S0 = 1000.0
        D = 0.001

        [Fitting.solver.bounds]
        S0 = [1.0, 5000.0]
        D = [1e-5, 0.1]
    """
        )
    )
    return path


def _write_nnls_config(path: Path, n_bins: int = _NNLS_N_BINS) -> Path:
    """Write a minimal NNLS pixelwise TOML config to *path*."""
    path.write_text(
        textwrap.dedent(
            f"""\
        [Fitting]
        fitter = "pixelwise"

        [Fitting.model]
        type = "nnls"
        d_range = [0.0001, 0.3]
        n_bins = {n_bins}

        [Fitting.solver]
        type = "nnls"
        reg_order = 0
        mu = 0.0
        max_iter = 100
        tol = 1e-6
        multi_threading = false
    """
        )
    )
    return path


# ---------------------------------------------------------------------------
# _reconstruct_maps
# ---------------------------------------------------------------------------


class TestReconstructMaps:
    """Unit tests for the _reconstruct_maps spatial reconstruction helper."""

    @pytest.mark.unit
    def test_correct_value_at_pixel_position(self):
        """Fitted value appears at the correct spatial location."""
        maps = _reconstruct_maps(
            {"S0": np.array([100.0, 200.0, 300.0])},
            [(0, 0, 0), (1, 1, 0), (2, 2, 0)],
            (3, 3, 1),
        )
        assert maps["S0"][0, 0, 0] == pytest.approx(100.0)
        assert maps["S0"][1, 1, 0] == pytest.approx(200.0)
        assert maps["S0"][2, 2, 0] == pytest.approx(300.0)

    @pytest.mark.unit
    def test_unvisited_voxels_are_zero(self):
        """Voxels not in pixel_indices are zero-filled."""
        maps = _reconstruct_maps(
            {"S0": np.array([500.0])},
            [(0, 0, 0)],
            (4, 4, 1),
        )
        assert maps["S0"][1, 1, 0] == pytest.approx(0.0)
        assert maps["S0"][3, 3, 0] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_output_shape_matches_spatial_shape(self):
        """Each output volume has exactly the requested spatial_shape."""
        spatial_shape = (5, 6, 2)
        maps = _reconstruct_maps(
            {"S0": np.array([1.0, 2.0]), "D": np.array([0.001, 0.002])},
            [(0, 0, 0), (1, 2, 1)],
            spatial_shape,
        )
        assert maps["S0"].shape == spatial_shape
        assert maps["D"].shape == spatial_shape

    @pytest.mark.unit
    def test_all_parameter_keys_present(self):
        """All keys in fitted_params are present in the output dict."""
        maps = _reconstruct_maps(
            {"S0": np.array([1000.0]), "D": np.array([0.001])},
            [(0, 0, 0)],
            (2, 2, 1),
        )
        assert set(maps.keys()) == {"S0", "D"}

    @pytest.mark.unit
    def test_output_dtype_is_float32(self):
        """Reconstructed maps use float32 to keep NIfTI files compact."""
        maps = _reconstruct_maps(
            {"S0": np.array([1000.0])},
            [(0, 0, 0)],
            (2, 2, 1),
        )
        assert maps["S0"].dtype == np.float32

    @pytest.mark.unit
    def test_2d_values_produce_4d_volume(self):
        """2-D per-pixel values (n_pixels, n_bins) produce a 4-D spatial volume."""
        n_pixels, n_bins = 2, 5
        coeffs = np.arange(n_pixels * n_bins, dtype=float).reshape(n_pixels, n_bins)
        maps = _reconstruct_maps(
            {"coefficients": coeffs},
            [(0, 0, 0), (1, 1, 0)],
            (3, 3, 1),
        )
        assert maps["coefficients"].shape == (3, 3, 1, n_bins)

    @pytest.mark.unit
    def test_2d_values_correct_position(self):
        """Each row of a 2-D per-pixel array lands at the correct voxel."""
        coeffs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        maps = _reconstruct_maps(
            {"coefficients": coeffs},
            [(0, 0, 0), (1, 1, 0)],
            (2, 2, 1),
        )
        np.testing.assert_array_equal(maps["coefficients"][0, 0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(maps["coefficients"][1, 1, 0], [4.0, 5.0, 6.0])

    @pytest.mark.unit
    def test_2d_unvisited_voxels_are_zero(self):
        """Unvisited voxels in a 4-D coefficients map are all-zero vectors."""
        coeffs = np.ones((1, 4))
        maps = _reconstruct_maps(
            {"coefficients": coeffs},
            [(0, 0, 0)],
            (3, 3, 1),
        )
        np.testing.assert_array_equal(maps["coefficients"][2, 2, 0], np.zeros(4))


# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Unit tests for the CLI argument parser."""

    @pytest.mark.unit
    def test_image_is_required(self):
        """Omitting --image raises SystemExit."""
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--bval", "x.bval", "--config", "x.toml"])

    @pytest.mark.unit
    def test_bval_is_required(self):
        """Omitting --bval raises SystemExit."""
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--image", "x.nii.gz", "--config", "x.toml"])

    @pytest.mark.unit
    def test_config_is_required(self):
        """Omitting --config raises SystemExit."""
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--image", "x.nii.gz", "--bval", "x.bval"])

    @pytest.mark.unit
    def test_seg_defaults_to_none(self):
        """--seg defaults to None when omitted."""
        args = _build_parser().parse_args(
            ["--image", "x.nii.gz", "--bval", "x.bval", "--config", "x.toml"]
        )
        assert args.seg is None

    @pytest.mark.unit
    def test_output_defaults_to_none(self):
        """--output defaults to None when omitted."""
        args = _build_parser().parse_args(
            ["--image", "x.nii.gz", "--bval", "x.bval", "--config", "x.toml"]
        )
        assert args.output is None

    @pytest.mark.unit
    def test_verbose_defaults_false(self):
        """--verbose defaults to False when omitted."""
        args = _build_parser().parse_args(
            ["--image", "x.nii.gz", "--bval", "x.bval", "--config", "x.toml"]
        )
        assert args.verbose is False

    @pytest.mark.unit
    def test_verbose_flag_sets_true(self):
        """Providing --verbose sets verbose to True."""
        args = _build_parser().parse_args(
            [
                "--image",
                "x.nii.gz",
                "--bval",
                "x.bval",
                "--config",
                "x.toml",
                "--verbose",
            ]
        )
        assert args.verbose is True

    @pytest.mark.unit
    def test_short_flags_accepted(self):
        """Short flags -i / -b / -c are recognised as aliases."""
        args = _build_parser().parse_args(
            ["-i", "img.nii.gz", "-b", "b.bval", "-c", "cfg.toml"]
        )
        assert str(args.image) == "img.nii.gz"
        assert str(args.bval) == "b.bval"
        assert str(args.config) == "cfg.toml"

    @pytest.mark.unit
    def test_seg_short_flag_accepted(self):
        """Short flag -s is recognised as alias for --seg."""
        args = _build_parser().parse_args(
            ["-i", "i.nii.gz", "-b", "b.bval", "-c", "c.toml", "-s", "seg.nii.gz"]
        )
        assert str(args.seg) == "seg.nii.gz"

    @pytest.mark.unit
    def test_output_short_flag_accepted(self):
        """Short flag -o is recognised as alias for --output."""
        args = _build_parser().parse_args(
            ["-i", "i.nii.gz", "-b", "b.bval", "-c", "c.toml", "-o", "results"]
        )
        assert str(args.output) == "results"


# ---------------------------------------------------------------------------
# main() — error paths
# ---------------------------------------------------------------------------


class TestCliMainErrors:
    """Tests for main() exit codes when inputs are missing or malformed."""

    @pytest.mark.unit
    def test_missing_image_returns_2(self, tmp_path):
        """main() returns exit code 2 when the image file does not exist."""
        code = main(
            [
                "--image",
                str(tmp_path / "missing.nii.gz"),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
            ]
        )
        assert code == 2

    @pytest.mark.unit
    def test_missing_bval_returns_2(self, tmp_path):
        """main() returns exit code 2 when the bval file does not exist."""
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(tmp_path / "missing.bval"),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
            ]
        )
        assert code == 2

    @pytest.mark.unit
    def test_missing_config_returns_2(self, tmp_path):
        """main() returns exit code 2 when the config file does not exist."""
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(tmp_path / "missing.toml"),
            ]
        )
        assert code == 2

    @pytest.mark.unit
    def test_unknown_model_type_returns_1(self, tmp_path):
        """main() returns exit code 1 when the config specifies an unknown model."""
        bad_cfg = tmp_path / "bad.toml"
        bad_cfg.write_text(
            textwrap.dedent(
                """\
            [Fitting]
            fitter = "pixelwise"
            [Fitting.model]
            type = "unknownmodel"
            [Fitting.solver]
            type = "curvefit"
            max_iter = 10
            tol = 1e-6
            [Fitting.solver.p0]
            S0 = 1000.0
            [Fitting.solver.bounds]
            S0 = [1.0, 5000.0]
        """
            )
        )
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(bad_cfg),
            ]
        )
        assert code == 1

    @pytest.mark.unit
    def test_config_missing_fitting_section_returns_1(self, tmp_path):
        """main() returns exit code 1 when [Fitting] section is absent."""
        bad_cfg = tmp_path / "no_fitting.toml"
        bad_cfg.write_text("[Other]\nkey = 1\n")
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(bad_cfg),
            ]
        )
        assert code == 1


# ---------------------------------------------------------------------------
# main() — success paths
# ---------------------------------------------------------------------------


class TestCliMainSuccess:
    """Integration tests for main() with valid inputs."""

    @pytest.mark.integration
    def test_returns_exit_code_0(self, tmp_path):
        """main() returns exit code 0 for valid image, bval, and config."""
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert code == 0

    @pytest.mark.integration
    def test_creates_one_nifti_per_parameter(self, tmp_path):
        """main() writes one .nii.gz file per model parameter."""
        out_dir = tmp_path / "out"
        main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
                "--output",
                str(out_dir),
            ]
        )
        assert (out_dir / "dwi_S0.nii.gz").exists()
        assert (out_dir / "dwi_D.nii.gz").exists()

    @pytest.mark.integration
    def test_output_map_spatial_shape_matches_image(self, tmp_path):
        """Saved S0 map has the same spatial shape as the input DWI image."""
        out_dir = tmp_path / "out"
        main(
            [
                "--image",
                str(
                    _write_nifti(
                        tmp_path / "dwi.nii.gz", _make_dwi(n_x=4, n_y=4, n_z=1)
                    )
                ),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
                "--output",
                str(out_dir),
            ]
        )
        s0 = nib.load(str(out_dir / "dwi_S0.nii.gz"))  # type: ignore
        assert s0.shape[:3] == (4, 4, 1)  # type: ignore

    @pytest.mark.integration
    def test_output_defaults_to_image_parent(self, tmp_path):
        """When --output is omitted, maps are written in the same dir as the image."""
        main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
            ]
        )
        assert (tmp_path / "dwi_S0.nii.gz").exists()
        assert (tmp_path / "dwi_D.nii.gz").exists()

    @pytest.mark.integration
    def test_with_segmentation_succeeds(self, tmp_path):
        """main() returns exit code 0 and creates outputs when --seg is provided."""
        seg = np.zeros((4, 4, 1), dtype=np.uint8)
        seg[1:3, 1:3, 0] = 1
        out_dir = tmp_path / "out"
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
                "--seg",
                str(_write_nifti(tmp_path / "seg.nii.gz", seg)),
                "--output",
                str(out_dir),
            ]
        )
        assert code == 0
        assert (out_dir / "dwi_S0.nii.gz").exists()

    @pytest.mark.integration
    def test_verbose_flag_does_not_affect_outputs(self, tmp_path):
        """--verbose produces the same output files and exit code as without it."""
        out_dir = tmp_path / "out"
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
                "--output",
                str(out_dir),
                "--verbose",
            ]
        )
        assert code == 0
        assert (out_dir / "dwi_S0.nii.gz").exists()

    @pytest.mark.integration
    def test_stem_stripping_nii_gz(self, tmp_path):
        """Output filenames strip .nii.gz from the image stem correctly."""
        out_dir = tmp_path / "out"
        main(
            [
                "--image",
                str(_write_nifti(tmp_path / "subject01.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_monoexp_config(tmp_path / "cfg.toml")),
                "--output",
                str(out_dir),
            ]
        )
        assert (out_dir / "subject01_S0.nii.gz").exists()
        assert (out_dir / "subject01_D.nii.gz").exists()


# ---------------------------------------------------------------------------
# main() — NNLS model + solver
# ---------------------------------------------------------------------------


class TestCliMainNNLS:
    """Integration tests for main() using NNLSModel and NNLSSolver."""

    @pytest.mark.integration
    def test_nnls_returns_exit_code_0(self, tmp_path):
        """main() returns exit code 0 for a valid NNLS configuration."""
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_nnls_config(tmp_path / "cfg.toml")),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert code == 0

    @pytest.mark.integration
    def test_nnls_creates_coefficients_nifti(self, tmp_path):
        """main() writes a 'coefficients' NIfTI file when using NNLSSolver."""
        out_dir = tmp_path / "out"
        main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_nnls_config(tmp_path / "cfg.toml")),
                "--output",
                str(out_dir),
            ]
        )
        assert (out_dir / "dwi_coefficients.nii.gz").exists()

    @pytest.mark.integration
    def test_nnls_output_is_4d_with_n_bins_channels(self, tmp_path):
        """Coefficients NIfTI has shape (X, Y, Z, n_bins)."""
        out_dir = tmp_path / "out"
        main(
            [
                "--image",
                str(
                    _write_nifti(
                        tmp_path / "dwi.nii.gz", _make_dwi(n_x=3, n_y=3, n_z=1)
                    )
                ),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_nnls_config(tmp_path / "cfg.toml", n_bins=_NNLS_N_BINS)),
                "--output",
                str(out_dir),
            ]
        )
        img = nib.load(str(out_dir / "dwi_coefficients.nii.gz"))  # type: ignore
        assert img.shape == (3, 3, 1, _NNLS_N_BINS)  # type: ignore

    @pytest.mark.integration
    def test_nnls_coefficients_are_nonnegative(self, tmp_path):
        """All values in the NNLS coefficients map are >= 0."""
        out_dir = tmp_path / "out"
        main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_nnls_config(tmp_path / "cfg.toml")),
                "--output",
                str(out_dir),
            ]
        )
        coeffs = nib.load(str(out_dir / "dwi_coefficients.nii.gz")).get_fdata()  # type: ignore
        assert np.all(coeffs >= 0.0)

    @pytest.mark.integration
    def test_nnls_spatial_shape_matches_image(self, tmp_path):
        """The first three dimensions of the coefficients map match the input image."""
        out_dir = tmp_path / "out"
        main(
            [
                "--image",
                str(
                    _write_nifti(
                        tmp_path / "dwi.nii.gz", _make_dwi(n_x=5, n_y=3, n_z=2)
                    )
                ),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_nnls_config(tmp_path / "cfg.toml")),
                "--output",
                str(out_dir),
            ]
        )
        img = nib.load(str(out_dir / "dwi_coefficients.nii.gz"))  # type: ignore
        assert img.shape[:3] == (5, 3, 2)  # type: ignore

    @pytest.mark.integration
    def test_nnls_with_segmentation(self, tmp_path):
        """main() succeeds with --seg, fitting only the masked voxels."""
        seg = np.zeros((4, 4, 1), dtype=np.uint8)
        seg[1:3, 1:3, 0] = 1
        out_dir = tmp_path / "out"
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_nnls_config(tmp_path / "cfg.toml")),
                "--seg",
                str(_write_nifti(tmp_path / "seg.nii.gz", seg)),
                "--output",
                str(out_dir),
            ]
        )
        assert code == 0
        assert (out_dir / "dwi_coefficients.nii.gz").exists()


# ---------------------------------------------------------------------------
# --fixed argument parsing
# ---------------------------------------------------------------------------


class TestBuildParserFixed:
    """Unit tests for the --fixed CLI argument."""

    @pytest.mark.unit
    def test_fixed_defaults_to_none(self):
        """--fixed defaults to None when omitted."""
        args = _build_parser().parse_args(
            ["-i", "x.nii.gz", "-b", "x.bval", "-c", "x.toml"]
        )
        assert args.fixed is None

    @pytest.mark.unit
    def test_single_fixed_arg(self):
        """Single --fixed is stored as a one-element list."""
        args = _build_parser().parse_args(
            [
                "-i",
                "x.nii.gz",
                "-b",
                "x.bval",
                "-c",
                "x.toml",
                "--fixed",
                "T1:/path/to/t1.nii.gz",
            ]
        )
        assert args.fixed == ["T1:/path/to/t1.nii.gz"]

    @pytest.mark.unit
    def test_multiple_fixed_args(self):
        """Multiple --fixed flags accumulate into a list."""
        args = _build_parser().parse_args(
            [
                "-i",
                "x.nii.gz",
                "-b",
                "x.bval",
                "-c",
                "x.toml",
                "--fixed",
                "T1:t1.nii.gz",
                "--fixed",
                "S0:s0.nii.gz",
            ]
        )
        assert args.fixed == ["T1:t1.nii.gz", "S0:s0.nii.gz"]

    @pytest.mark.unit
    def test_short_flag_f(self):
        """Short flag -f works as alias for --fixed."""
        args = _build_parser().parse_args(
            ["-i", "x.nii.gz", "-b", "x.bval", "-c", "x.toml", "-f", "T1:t1.nii.gz"]
        )
        assert args.fixed == ["T1:t1.nii.gz"]


# ---------------------------------------------------------------------------
# main() with --fixed
# ---------------------------------------------------------------------------


def _write_t1_monoexp_config(path: Path) -> Path:
    """Write a monoexp + T1 TOML config (p0/bounds include T1)."""
    import textwrap

    path.write_text(
        textwrap.dedent(
            """\
        [Fitting]
        fitter = "pixelwise"

        [Fitting.model]
        type = "monoexp"
        fit_t1 = true
        repetition_time = 3000.0

        [Fitting.solver]
        type = "curvefit"
        max_iter = 500
        tol = 1e-10

        [Fitting.solver.p0]
        S0 = 900.0
        D = 0.001
        T1 = 1000.0

        [Fitting.solver.bounds]
        S0 = [1.0, 5000.0]
        D = [1e-5, 0.1]
        T1 = [100.0, 5000.0]
    """
        )
    )
    return path


def _make_t1_dwi(
    n_x: int = 4,
    n_y: int = 4,
    n_z: int = 1,
    S0: float = 1000.0,
    D: float = 0.001,
    T1: float = 1000.0,
) -> np.ndarray:
    """Synthetic noise-free monoexp DWI with T1 correction."""
    model = MonoExpModel(fit_t1=True, repetition_time=3000.0)
    signal = model.forward(B_VALUES, S0, D, T1)
    return np.tile(signal, (n_x, n_y, n_z, 1))


class TestCliMainFixed:
    """Integration tests for main() with --fixed per-pixel NIfTI maps."""

    @pytest.mark.integration
    def test_fixed_t1_map_runs_successfully(self, tmp_path):
        """main() succeeds with --fixed T1:path and produces S0 + D maps."""
        S0, D, T1 = 1000.0, 0.001, 1000.0
        dwi = _make_t1_dwi(n_x=2, n_y=2, S0=S0, D=D, T1=T1)
        t1_map = np.full((2, 2, 1), T1, dtype=np.float32)

        out_dir = tmp_path / "out"
        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", dwi)),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_t1_monoexp_config(tmp_path / "cfg.toml")),
                "--fixed",
                f"T1:{_write_nifti(tmp_path / 't1.nii.gz', t1_map)}",
                "--output",
                str(out_dir),
            ]
        )
        assert code == 0
        assert (out_dir / "dwi_S0.nii.gz").exists()
        assert (out_dir / "dwi_D.nii.gz").exists()
        # T1 was fixed, not fitted — no T1 map should be produced
        assert not (out_dir / "dwi_T1.nii.gz").exists()

    @pytest.mark.integration
    def test_fixed_spatial_mismatch_returns_1(self, tmp_path):
        """main() returns exit code 1 when fixed map shape != DWI shape."""
        dwi = _make_t1_dwi(n_x=4, n_y=4)
        bad_t1 = np.full((2, 2, 1), 1000.0, dtype=np.float32)  # wrong shape

        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", dwi)),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_t1_monoexp_config(tmp_path / "cfg.toml")),
                "--fixed",
                f"T1:{_write_nifti(tmp_path / 't1.nii.gz', bad_t1)}",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert code == 1

    @pytest.mark.integration
    def test_fixed_invalid_format_returns_1(self, tmp_path):
        """main() returns exit code 1 when --fixed lacks a colon separator."""
        dwi = _make_t1_dwi(n_x=2, n_y=2)

        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", dwi)),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_t1_monoexp_config(tmp_path / "cfg.toml")),
                "--fixed",
                "T1_no_colon",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert code == 1

    @pytest.mark.integration
    def test_fixed_nonexistent_nifti_returns_2(self, tmp_path):
        """main() returns exit code 2 when the fixed param NIfTI doesn't exist."""
        dwi = _make_t1_dwi(n_x=2, n_y=2)

        code = main(
            [
                "--image",
                str(_write_nifti(tmp_path / "dwi.nii.gz", dwi)),
                "--bval",
                str(_write_bval(tmp_path / "b.bval")),
                "--config",
                str(_write_t1_monoexp_config(tmp_path / "cfg.toml")),
                "--fixed",
                "T1:/nonexistent/t1_map.nii.gz",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert code == 2
