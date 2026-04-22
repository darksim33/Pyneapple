from __future__ import annotations

# --- IMPORTS ---
# Fitters
from .fitters import (
    IDEALFitter,
    PixelWiseFitter,
    SegmentationWiseFitter,
    SegmentedFitter,
)

# Models
from .models import (
    BiExpModel,
    MonoExpModel,
    NNLSModel,
    TriExpModel,
)

# Solvers
from .solvers import (
    ConstrainedCurveFitSolver,
    CurveFitSolver,
    NNLSSolver,
)

__all__ = [
    # Models
    "MonoExpModel",
    "BiExpModel",
    "TriExpModel",
    "NNLSModel",
    # Solvers
    "CurveFitSolver",
    "ConstrainedCurveFitSolver",
    "NNLSSolver",
    # Fitters
    "PixelWiseFitter",
    "SegmentationWiseFitter",
    "IDEALFitter",
    "SegmentedFitter",
]


# --- LOGGING CONFIGURATION ---
import sys

from loguru import logger


def configure_logging(level: str = "INFO", **kwargs):
    """
    Configure loguru logging for the package.

    Parameters
    ----------
    level : str, optional
        Logging level. Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        Default is 'INFO'.
    **kwargs : dict
        Additional keyword arguments passed to logger.configure().

    Examples
    --------
    >>> import pyneapple
    >>> pyneapple.configure_logging(level='DEBUG')
    >>> pyneapple.configure_logging(level='WARNING', format="{time} - {message}")
    """
    # Remove default handler
    logger.remove()

    # Add new handler with specified configuration
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        **kwargs,
    )


# Configure default logging
configure_logging(level="WARNING")
