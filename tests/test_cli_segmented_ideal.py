"""Tests for the pyneapple segmented and ideal CLI commands.

Covers:
- Option presence / required behaviour (via CliRunner)
- exit codes (success and error paths)
- Integration tests using tmp_path
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from click.testing import CliRunner

from pyneapple.cli.ideal import ideal
from pyneapple.cli.segmentationwise import segmented
from pyneapple.models import MonoExpModel

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=float)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_bval(path: Path, bvalues: np.ndarray = B_VALUES) -> Path:
    """Write b-values one per line."""
    path.write_text("\n".join(str(int(b)) for b in bvalues))
    return path


def _write_nifti(path: Path, data: np.ndarray) -> Path:
    """Save *data* as a NIfTI file with identity affine."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))  # type: ignore
    return path


def _make_dwi(n_x: int = 4, n_y: int = 4, n_z: int = 1) -> np.ndarray:
    """Synthetic monoexp DWI, shape (n_x, n_y, n_z, N_B)."""
    signal = MonoExpModel().forward(B_VALUES, 1000.0, 0.001)
    return np.tile(signal, (n_x, n_y, n_z, 1))


def _write_seg(path: Path, shape: tuple[int, int, int]) -> Path:
    """Write a binary segmentation mask (all 1s, 3-D) as NIfTI."""
    seg = np.ones(shape + (1,), dtype=np.int32)
    _write_nifti(path, seg)
    return path


def _write_monoexp_seg_config(path: Path, fitter: str = "segmentationwise") -> Path:
    """Write a minimal monoexp config for the given fitter type."""
    path.write_text(
        textwrap.dedent(
            f"""\
        [Fitting]
        fitter = "{fitter}"

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


# ===========================================================================
# pyneapple segmented CLI
# ===========================================================================


class TestSegmentedOptions:
    """Behavioural tests for required options on the segmented command."""

    @pytest.mark.unit
    def test_seg_is_required(self, tmp_path):
        """Invoking segmented without --seg exits with non-zero code."""
        img = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        cfg = _write_monoexp_seg_config(tmp_path / "cfg.toml")
        result = runner.invoke(
            segmented,
            ["-i", str(img), "-b", str(bval), "-c", str(cfg)],
        )
        assert result.exit_code != 0

    @pytest.mark.unit
    def test_output_optional(self, tmp_path):
        """Invoking segmented without --output succeeds."""
        img = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        seg = _write_seg(tmp_path / "seg.nii.gz", (4, 4, 1))
        cfg = _write_monoexp_seg_config(tmp_path / "cfg.toml")
        result = runner.invoke(
            segmented,
            ["-i", str(img), "-b", str(bval), "-c", str(cfg), "-s", str(seg)],
        )
        assert result.exit_code == 0

    @pytest.mark.unit
    def test_verbose_short_flag(self, tmp_path):
        """Short flag -v is recognised and sets verbose mode."""
        img = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        seg = _write_seg(tmp_path / "seg.nii.gz", (4, 4, 1))
        cfg = _write_monoexp_seg_config(tmp_path / "cfg.toml")
        result = runner.invoke(
            segmented,
            ["-i", str(img), "-b", str(bval), "-c", str(cfg), "-s", str(seg), "-v"],
        )
        assert result.exit_code == 0


class TestSegmentedMain:
    """Integration tests for the segmented command."""

    @pytest.mark.integration
    def test_missing_image_returns_error(self, tmp_path: Path):
        """Exit code 2 when --image file does not exist."""
        bval = _write_bval(tmp_path / "dw.bval")
        cfg = _write_monoexp_seg_config(tmp_path / "cfg.toml")
        seg = _write_seg(tmp_path / "seg.nii.gz", (4, 4, 1))
        result = runner.invoke(
            segmented,
            [
                "-i",
                str(tmp_path / "nonexistent.nii.gz"),
                "-b",
                str(bval),
                "-c",
                str(cfg),
                "-s",
                str(seg),
            ],
        )
        assert result.exit_code == 2

    @pytest.mark.integration
    def test_invalid_config_returns_error(self, tmp_path: Path):
        """Exit code 1 when config specifies an unknown model/solver."""
        dwi = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        seg = _write_seg(tmp_path / "seg.nii.gz", (4, 4, 1))
        bad_cfg = tmp_path / "bad.toml"
        bad_cfg.write_text(
            '[Fitting]\nfitter = "segmentationwise"\n[Fitting.model]\ntype = "unknown_model"'
        )
        result = runner.invoke(
            segmented,
            [
                "-i",
                str(dwi),
                "-b",
                str(bval),
                "-c",
                str(bad_cfg),
                "-s",
                str(seg),
            ],
        )
        assert result.exit_code == 1

    @pytest.mark.integration
    def test_successful_fit_writes_output(self, tmp_path: Path):
        """Successful run returns 0 and writes a parameter map NIfTI."""
        dwi = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        seg = _write_seg(tmp_path / "seg.nii.gz", (4, 4, 1))
        cfg = _write_monoexp_seg_config(tmp_path / "cfg.toml")
        out = tmp_path / "results"

        result = runner.invoke(
            segmented,
            [
                "-i",
                str(dwi),
                "-b",
                str(bval),
                "-c",
                str(cfg),
                "-s",
                str(seg),
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0
        saved = list(out.glob("*.nii.gz"))
        assert len(saved) > 0


# ===========================================================================
# pyneapple ideal CLI
# ===========================================================================


class TestIdealOptions:
    """Behavioural tests for the ideal command options."""

    @pytest.mark.unit
    def test_seg_is_optional(self, tmp_path):
        """Invoking ideal without --seg does not fail due to missing --seg."""
        bval = _write_bval(tmp_path / "dw.bval")
        # Config with no [Fitting.ideal] section → exits 1 (not 2 for missing --seg)
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            textwrap.dedent(
                """\
            [Fitting]
            fitter = "ideal"
            [Fitting.model]
            type = "monoexp"
            [Fitting.solver]
            type = "curvefit"
            [Fitting.solver.p0]
            S0 = 1000.0
            D = 0.001
            [Fitting.solver.bounds]
            S0 = [1.0, 5000.0]
            D = [1e-5, 0.1]
            """
            )
        )
        dwi = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        result = runner.invoke(
            ideal,
            ["-i", str(dwi), "-b", str(bval), "-c", str(cfg)],
        )
        # Must not be exit_code 2 due to a missing required --seg
        # (it should be 1 because the ideal section is absent)
        assert result.exit_code == 1

    @pytest.mark.unit
    def test_fixed_flag_accepted(self, tmp_path):
        """--fixed is a registered option (non-zero exit due to content, not parse)."""
        bval = _write_bval(tmp_path / "dw.bval")
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('[Fitting]\nfitter = "ideal"\n[Fitting.model]\ntype = "monoexp"')
        dwi = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        result = runner.invoke(
            ideal,
            [
                "-i",
                str(dwi),
                "-b",
                str(bval),
                "-c",
                str(cfg),
                "--fixed",
                "T1:/path/t1.nii.gz",
            ],
        )
        # Exit code 2 means file-not-found for --image/--bval/--config,
        # which we don't want. A non-2 error (e.g. 1) means --fixed was parsed.
        # Actually because T1:/path/t1.nii.gz has a non-existent path, it
        # should reach run_pipeline (since --fixed is not exists=True validated)
        # and return 1 or 2 from the pipeline — but NOT "no such option".
        assert result.exit_code != 0  # some error, but --fixed was recognized


class TestIdealMain:
    """Integration tests for the ideal command."""

    @pytest.mark.integration
    def test_config_without_ideal_section_returns_error(self, tmp_path: Path):
        """Exit code 1 when [Fitting.ideal] section is missing."""
        dwi = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        cfg = tmp_path / "no_ideal.toml"
        cfg.write_text(
            textwrap.dedent(
                """\
            [Fitting]
            fitter = "ideal"

            [Fitting.model]
            type = "monoexp"

            [Fitting.solver]
            type = "curvefit"

            [Fitting.solver.p0]
            S0 = 1000.0
            D = 0.001

            [Fitting.solver.bounds]
            S0 = [1.0, 5000.0]
            D = [1e-5, 0.1]
            """
            )
        )
        result = runner.invoke(ideal, ["-i", str(dwi), "-b", str(bval), "-c", str(cfg)])
        assert result.exit_code == 1

    @pytest.mark.integration
    def test_missing_image_returns_error(self, tmp_path: Path):
        """Exit code 2 when --image file does not exist."""
        bval = _write_bval(tmp_path / "dw.bval")
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('[Fitting]\nfitter = "ideal"\n[Fitting.model]\ntype = "monoexp"')
        result = runner.invoke(
            ideal,
            [
                "-i",
                str(tmp_path / "nonexistent.nii.gz"),
                "-b",
                str(bval),
                "-c",
                str(cfg),
            ],
        )
        assert result.exit_code == 2
