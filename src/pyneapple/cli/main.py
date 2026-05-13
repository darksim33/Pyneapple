"""Unified Pyneapple CLI group.

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

import importlib.metadata
import sys

import click

from .ideal import ideal
from .pixelwise import pixelwise
from .segmentationwise import segmented

# ---------------------------------------------------------------------------
# Info helper (plain function — not Click-decorated so tests can call it
# directly with capsys)
# ---------------------------------------------------------------------------


def _info() -> None:
    """Print version, available models, solvers, and fitters."""
    try:
        version = importlib.metadata.version("pyneapple")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (package not installed)"

    from ..fitters import _REGISTRY as _FITTER_REG
    from ..models import _REGISTRY as _MODEL_REG
    from ..solvers import _REGISTRY as _SOLVER_REG

    print(f"Pyneapple {version}")
    print(f"Python    {sys.version.split()[0]}")
    print()
    print(f"Models  : {', '.join(sorted(_MODEL_REG))}")
    print(f"Solvers : {', '.join(sorted(_SOLVER_REG))}")
    print(f"Fitters : {', '.join(sorted(_FITTER_REG))}")


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group(
    invoke_without_command=True,
    epilog=(
        "Examples:\n\n"
        "  pyneapple pixelwise -i dwi.nii.gz -b dwi.bval -c monoexp.toml\n\n"
        "  pyneapple segmented -i dwi.nii.gz -b dwi.bval -c biexp.toml "
        "-s mask.nii.gz\n\n"
        "  pyneapple ideal -i dwi.nii.gz -b dwi.bval -c ideal.toml\n\n"
        "  pyneapple info"
    ),
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Pyneapple — multi-exponential DWI diffusion fitting toolkit."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("info")
def info_cmd() -> None:
    """Print version and available components."""
    _info()


cli.add_command(pixelwise)
cli.add_command(segmented)
cli.add_command(ideal)


if __name__ == "__main__":
    cli()
