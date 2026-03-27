"""Command-line interface package for Pyneapple."""

from .pixelwise import main as pixelwise_main
from .segmentationwise import main as segmented_main
from .ideal import main as ideal_main
from .main import main as dispatch_main

__all__ = [
    "pixelwise_main",
    "segmented_main",
    "ideal_main",
    "dispatch_main",
]
