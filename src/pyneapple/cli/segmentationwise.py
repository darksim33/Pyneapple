"""CLI command: segmentation-wise diffusion MRI fitting.

Registered on the ``pyneapple`` group as ``pyneapple segmented``.
``--seg`` is **required** for this command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ._common import shared_options, run_pipeline


@click.command("segmented")
@shared_options
@click.option(
    "--seg",
    "-s",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    metavar="PATH",
    help="Segmentation mask NIfTI (.nii / .nii.gz) — required.",
)
def segmented(
    image: Path,
    bval: Path,
    config: Path,
    output: Path | None,
    verbose: bool,
    fixed: tuple[str, ...],
    diagnostics: bool,
    seg: Path,
) -> None:
    """Fit the mean signal per labelled ROI (--seg required).

    Fits the mean signal of each labelled region in the segmentation mask
    and writes one NIfTI parameter map per fitted parameter.
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
