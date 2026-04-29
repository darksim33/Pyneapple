"""CLI entry point: IDEAL diffusion MRI fitting.

Usage
-----
::

    pyneapple-ideal \\
        --image  dwi.nii.gz \\
        --bval   dwi.bval \\
        --config config.toml \\
        --seg    mask.nii.gz \\
        [--output ./results] \\
        [--verbose]

The TOML config **must** include a ``[Fitting.ideal]`` section that specifies
at least ``dim_steps`` and ``step_tol``.  Example::

    [Fitting.ideal]
    dim_steps = [[16, 16], [32, 32], [64, 64], [128, 128]]
    ideal_dims = 2
    segmentation_threshold = 0.2
    interpolation_method   = "cubic"

    [Fitting.ideal.step_tol]
    S0 = 0.5
    f1 = 0.2
    D1 = 0.2
    D2 = 0.2
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ._common import add_shared_args, run_pipeline


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyneapple-ideal",
        description=(
            "IDEAL diffusion MRI fitting with Pyneapple. "
            "Iteratively refines parameter maps on a multi-resolution grid. "
            "IDEAL parameters are read from the [Fitting.ideal] TOML section."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_args(parser, seg_required=False)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        int: Exit code: 0 on success, non-zero on failure.
    """
    args = _build_parser().parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
