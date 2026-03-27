"""CLI entry point: segmentation-wise diffusion MRI fitting.

Usage
-----
::

    pyneapple-segmented \\
        --image  dwi.nii.gz \\
        --bval   dwi.bval \\
        --config config.toml \\
        --seg    mask.nii.gz \\
        [--output ./results] \\
        [--verbose]

Fits the mean signal of each labelled ROI in ``--seg`` and writes one NIfTI
parameter map per fitted parameter.  The segmentation mask is **required**
for this fitting mode.
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
        prog="pyneapple-segmented",
        description=(
            "Segmentation-wise diffusion MRI fitting with Pyneapple. "
            "Fits the mean signal within each ROI of the segmentation mask."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_shared_args(parser, seg_required=True)
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
