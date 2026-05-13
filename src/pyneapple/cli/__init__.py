"""Command-line interface package for Pyneapple."""

from .ideal import ideal
from .main import cli
from .pixelwise import pixelwise
from .segmentationwise import segmented

__all__ = [
    "pixelwise",
    "segmented",
    "ideal",
    "cli",
]
