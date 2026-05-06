"""Tests for the unified pyneapple dispatch CLI (main.py).

Covers:
- Subcommand routing (pixelwise, segmented, ideal, info)
- pyneapple info output
- No-args case (prints help)
- Invalid subcommand
- --help for each subcommand
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from pyneapple.cli.main import cli, _info


runner = CliRunner()


# ---------------------------------------------------------------------------
# _info()  (plain function — tested with capsys)
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
# cli group dispatch
# ---------------------------------------------------------------------------


class TestDispatchCli:
    """Tests for the cli Click group."""

    @pytest.mark.unit
    def test_info_subcommand_returns_zero(self):
        """'pyneapple info' exits with code 0."""
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0

    @pytest.mark.unit
    def test_no_args_returns_zero(self):
        """'pyneapple' with no arguments prints help and exits 0."""
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "pixelwise" in result.output or "usage" in result.output.lower()

    @pytest.mark.unit
    def test_pixelwise_help_exits_zero(self):
        """'pyneapple pixelwise --help' exits with code 0."""
        result = runner.invoke(cli, ["pixelwise", "--help"])
        assert result.exit_code == 0
        assert "--image" in result.output

    @pytest.mark.unit
    def test_segmented_help_exits_zero(self):
        """'pyneapple segmented --help' exits with code 0."""
        result = runner.invoke(cli, ["segmented", "--help"])
        assert result.exit_code == 0
        assert "--seg" in result.output

    @pytest.mark.unit
    def test_ideal_help_exits_zero(self):
        """'pyneapple ideal --help' exits with code 0."""
        result = runner.invoke(cli, ["ideal", "--help"])
        assert result.exit_code == 0
        assert "--image" in result.output

    @pytest.mark.unit
    def test_invalid_subcommand_exits_nonzero(self):
        """Unknown subcommand exits with non-zero code."""
        result = runner.invoke(cli, ["unknowncommand"])
        assert result.exit_code != 0
