"""CLI entry point: pixelwise diffusion MRI fitting.

Usage
-----
::

    pyneapple-pixelwise \\
        --image  dwi.nii.gz \\
        --bval   dwi.bval \\
        --config config.toml \\
        [--seg   mask.nii.gz] \\
        [--output ./results] \\
        [--verbose]

Outputs one NIfTI parameter map per fitted parameter, named
``<image_stem>_<param>.nii.gz`` in the chosen output directory.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ._common import (
    add_shared_args,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyneapple-pixelwise",
        description="Pixelwise diffusion MRI fitting with Pyneapple.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_args(parser, seg_required=False)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    args = _build_parser().parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
