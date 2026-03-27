"""Unified Pyneapple CLI dispatcher.

Usage
-----
::

    pyneapple <command> [options]

Commands
--------
pixelwise
    Fit each voxel independently.
segmented
    Fit the mean signal per ROI (requires ``--seg``).
ideal
    IDEAL iterative multi-resolution fitting.
info
    Print version and available components.

Run ``pyneapple <command> --help`` for per-command help.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from typing import Sequence

from .pixelwise import main as pixelwise_main
from .segmentationwise import main as segmented_main
from .ideal import main as ideal_main


# ---------------------------------------------------------------------------
# Info subcommand
# ---------------------------------------------------------------------------


def _info() -> None:
    """Print version, available models, solvers, and fitters."""
    try:
        version = importlib.metadata.version("pyneapple")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (package not installed)"

    from ..models import _REGISTRY as _model_reg
    from ..solvers import _REGISTRY as _solver_reg
    from ..fitters import _REGISTRY as _fitter_reg

    print(f"Pyneapple {version}")
    print(f"Python    {sys.version.split()[0]}")
    print()
    print(f"Models  : {', '.join(sorted(_model_reg))}")
    print(f"Solvers : {', '.join(sorted(_solver_reg))}")
    print(f"Fitters : {', '.join(sorted(_fitter_reg))}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyneapple",
        description="Pyneapple — multi-exponential DWI diffusion fitting toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pyneapple pixelwise  -i dwi.nii.gz -b dwi.bval -c monoexp.toml\n"
            "  pyneapple segmented  -i dwi.nii.gz -b dwi.bval -c biexp.toml   -s mask.nii.gz\n"
            "  pyneapple ideal      -i dwi.nii.gz -b dwi.bval -c ideal.toml   -s mask.nii.gz\n"
            "  pyneapple info\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    # --- pixelwise ---
    subparsers.add_parser(
        "pixelwise",
        help="Fit each voxel independently.",
        add_help=False,
    )

    # --- segmented ---
    subparsers.add_parser(
        "segmented",
        help="Fit mean signal per ROI (--seg required).",
        add_help=False,
    )

    # --- ideal ---
    subparsers.add_parser(
        "ideal",
        help="IDEAL iterative multi-resolution fitting.",
        add_help=False,
    )

    # --- info ---
    subparsers.add_parser(
        "info",
        help="Print version and available components.",
        add_help=False,
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Unified CLI entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    # Parse only the subcommand name; pass the remainder to the sub-main.
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    parser = _build_parser()

    # Show top-level help when invoked with no arguments
    if not argv:
        parser.print_help()
        return 0

    # We parse only the first positional argument (the subcommand).
    # Everything after it is forwarded verbatim so that sub-parsers handle
    # their own --help flags correctly.
    known, remainder = parser.parse_known_args(argv[:1])

    command = known.command
    sub_argv = argv[1:]  # arguments after the subcommand name

    if command == "pixelwise":
        return pixelwise_main(sub_argv)
    elif command == "segmented":
        return segmented_main(sub_argv)
    elif command == "ideal":
        return ideal_main(sub_argv)
    elif command == "info":
        _info()
        return 0

    # Unreachable — argparse enforces valid subcommands
    parser.print_help()  # pragma: no cover
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
