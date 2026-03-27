"""Tests for the pyneapple-segmented and pyneapple-ideal CLI tools.

Covers:
- _build_parser() argument definitions and required flags
- main() exit codes (success and error paths)
- Integration tests using tmp_path
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pyneapple.models import MonoExpModel
from pyneapple.cli.segmentationwise import (
    _build_parser as seg_build_parser,
    main as seg_main,
)
from pyneapple.cli.ideal import _build_parser as ideal_build_parser, main as ideal_main

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=float)


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
# pyneapple-segmented CLI
# ===========================================================================


class TestSegmentedParser:
    """Unit tests for the segmented CLI argument parser."""

    @pytest.mark.unit
    def test_prog_name(self):
        """Parser prog attribute is set correctly."""
        parser = seg_build_parser()
        assert parser.prog == "pyneapple-segmented"

    @pytest.mark.unit
    def test_seg_is_required(self):
        """--seg is required for the segmented CLI."""
        parser = seg_build_parser()
        seg_action = next(
            a for a in parser._actions if "--seg" in getattr(a, "option_strings", [])
        )
        assert seg_action.required is True

    @pytest.mark.unit
    def test_output_optional(self):
        """--output is optional with default None."""
        parser = seg_build_parser()
        output_action = next(
            a for a in parser._actions if "--output" in getattr(a, "option_strings", [])
        )
        assert output_action.default is None

    @pytest.mark.unit
    def test_verbose_flag(self):
        """--verbose stores True when present."""
        parser = seg_build_parser()
        args = parser.parse_args(
            ["-i", "x.nii", "-b", "x.bval", "-c", "x.toml", "-s", "seg.nii", "-v"]
        )
        assert args.verbose is True


class TestSegmentedMain:
    """Integration tests for pyneapple-segmented main()."""

    @pytest.mark.integration
    def test_missing_image_returns_error(self, tmp_path: Path):
        """Exit code 2 when --image file does not exist."""
        bval = _write_bval(tmp_path / "dw.bval")
        cfg = _write_monoexp_seg_config(tmp_path / "cfg.toml")
        seg = _write_seg(tmp_path / "seg.nii.gz", (4, 4, 1))
        ret = seg_main(
            [
                "-i",
                str(tmp_path / "nonexistent.nii.gz"),
                "-b",
                str(bval),
                "-c",
                str(cfg),
                "-s",
                str(seg),
            ]
        )
        assert ret == 2

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
        ret = seg_main(
            [
                "-i",
                str(dwi),
                "-b",
                str(bval),
                "-c",
                str(bad_cfg),
                "-s",
                str(seg),
            ]
        )
        assert ret == 1

    @pytest.mark.integration
    def test_successful_fit_writes_output(self, tmp_path: Path):
        """Successful run returns 0 and writes a parameter map NIfTI."""
        dwi = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        seg = _write_seg(tmp_path / "seg.nii.gz", (4, 4, 1))
        cfg = _write_monoexp_seg_config(tmp_path / "cfg.toml")
        out = tmp_path / "results"

        ret = seg_main(
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
            ]
        )
        assert ret == 0
        saved = list(out.glob("*.nii.gz"))
        assert len(saved) > 0


# ===========================================================================
# pyneapple-ideal CLI
# ===========================================================================


class TestIdealParser:
    """Unit tests for the ideal CLI argument parser."""

    @pytest.mark.unit
    def test_prog_name(self):
        """Parser prog attribute is set correctly."""
        parser = ideal_build_parser()
        assert parser.prog == "pyneapple-ideal"

    @pytest.mark.unit
    def test_seg_is_optional(self):
        """--seg is optional for the ideal CLI (no required=True)."""

        parser = ideal_build_parser()
        seg_action = next(
            a for a in parser._actions if "--seg" in getattr(a, "option_strings", [])
        )
        assert not seg_action.required

    @pytest.mark.unit
    def test_parsed_args_have_fixed(self):
        """--fixed argument is registered."""
        parser = ideal_build_parser()
        args = parser.parse_args(
            [
                "-i",
                "x.nii",
                "-b",
                "x.bval",
                "-c",
                "x.toml",
                "--fixed",
                "T1:/path/t1.nii.gz",
            ]
        )
        assert args.fixed == ["T1:/path/t1.nii.gz"]


class TestIdealMain:
    """Integration tests for pyneapple-ideal main()."""

    @pytest.mark.integration
    def test_config_without_ideal_section_returns_error(self, tmp_path: Path):
        """Exit code 1 when [Fitting.ideal] section is missing."""
        dwi = _write_nifti(tmp_path / "dwi.nii.gz", _make_dwi())
        bval = _write_bval(tmp_path / "dw.bval")
        # Config declares fitter=ideal but has no [Fitting.ideal] section
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
        ret = ideal_main(["-i", str(dwi), "-b", str(bval), "-c", str(cfg)])
        assert ret == 1

    @pytest.mark.integration
    def test_missing_image_returns_error(self, tmp_path: Path):
        """Exit code 2 when --image file does not exist."""
        bval = _write_bval(tmp_path / "dw.bval")
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('[Fitting]\nfitter = "ideal"\n[Fitting.model]\ntype = "monoexp"')
        ret = ideal_main(
            [
                "-i",
                str(tmp_path / "nonexistent.nii.gz"),
                "-b",
                str(bval),
                "-c",
                str(cfg),
            ]
        )
        assert ret == 2
