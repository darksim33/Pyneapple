"""Tests for the unified pyneapple dispatch CLI (main.py).

Covers:
- Subcommand routing (pixelwise, segmented, ideal, info)
- pyneapple info output
- No-args case
- Invalid subcommand
"""

from __future__ import annotations


import pytest

from pyneapple.cli.main import main, _info


# ---------------------------------------------------------------------------
# _info()
# ---------------------------------------------------------------------------


class TestInfoOutput:
    """Tests for the _info() helper."""

    @pytest.mark.unit
    def test_info_prints_version(self, capsys):
        """_info() prints a version line."""
        _info()
        captured = capsys.readouterr()
        assert "Pyneapple" in captured.out

    @pytest.mark.unit
    def test_info_lists_models(self, capsys):
        """_info() lists at least biexp in models."""
        _info()
        captured = capsys.readouterr()
        assert "biexp" in captured.out

    @pytest.mark.unit
    def test_info_lists_solvers(self, capsys):
        """_info() lists at least curvefit in solvers."""
        _info()
        captured = capsys.readouterr()
        assert "curvefit" in captured.out

    @pytest.mark.unit
    def test_info_lists_fitters(self, capsys):
        """_info() lists at least pixelwise in fitters."""
        _info()
        captured = capsys.readouterr()
        assert "pixelwise" in captured.out


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


class TestDispatchMain:
    """Tests for the main() dispatcher function."""

    @pytest.mark.unit
    def test_info_subcommand_returns_zero(self):
        """'pyneapple info' returns 0."""
        ret = main(["info"])
        assert ret == 0

    @pytest.mark.unit
    def test_no_args_returns_zero(self, capsys):
        """'pyneapple' with no arguments prints help and returns 0."""
        ret = main([])
        assert ret == 0
        captured = capsys.readouterr()
        assert "pixelwise" in captured.out or "usage" in captured.out.lower()

    @pytest.mark.unit
    def test_pixelwise_help_forwards_correctly(self, capsys):
        """'pyneapple pixelwise --help' exits via SystemExit (argparse help)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["pixelwise", "--help"])
        assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_segmented_help_forwards_correctly(self, capsys):
        """'pyneapple segmented --help' exits via SystemExit (argparse help)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["segmented", "--help"])
        assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_ideal_help_forwards_correctly(self, capsys):
        """'pyneapple ideal --help' exits via SystemExit (argparse help)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["ideal", "--help"])
        assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_invalid_subcommand_exits_nonzero(self):
        """Unknown subcommand exits with non-zero code."""
        with pytest.raises(SystemExit) as exc_info:
            main(["unknowncommand"])
        assert exc_info.value.code != 0
