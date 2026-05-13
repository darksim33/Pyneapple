"""CLI command: pixelwise diffusion MRI fitting.

Registered on the ``pyneapple`` group as ``pyneapple pixelwise``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ._common import run_pipeline, shared_options


@click.command("pixelwise")
@shared_options
@click.option(
    "--seg",
    "-s",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    metavar="PATH",
    help="Optional segmentation mask NIfTI (.nii / .nii.gz).",
)
def pixelwise(
    image: Path,
    bval: Path,
    config: Path,
    output: Path | None,
    verbose: bool,
    fixed: tuple[str, ...],
    diagnostics: bool,
    seg: Path | None,
) -> None:
    """Fit each voxel independently.

    Outputs one NIfTI parameter map per fitted parameter, named
    <image_stem>_<param>.nii.gz in the chosen output directory.
    """
    sys.exit(
        run_pipeline(
            image=image,
            bval=bval,
            config=config,
            seg=seg,
            output=output,
            verbose=verbose,
            fixed=fixed,
            diagnostics=diagnostics,
        )
    )
