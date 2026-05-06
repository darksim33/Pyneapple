"""CLI command: IDEAL diffusion MRI fitting.

Registered on the ``pyneapple`` group as ``pyneapple ideal``.

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

import sys
from pathlib import Path

import click

from ._common import shared_options, run_pipeline


@click.command("ideal")
@shared_options
@click.option(
    "--seg",
    "-s",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    metavar="PATH",
    help="Optional segmentation mask NIfTI (.nii / .nii.gz).",
)
def ideal(
    image: Path,
    bval: Path,
    config: Path,
    output: Path | None,
    verbose: bool,
    fixed: tuple[str, ...],
    seg: Path | None,
) -> None:
    """IDEAL iterative multi-resolution fitting.

    Iteratively refines parameter maps on a multi-resolution grid.
    IDEAL parameters are read from the [Fitting.ideal] TOML section.
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
        )
    )
