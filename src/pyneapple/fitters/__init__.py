from __future__ import annotations

from .base import BaseFitter
from .pixelwise import PixelWiseFitter
from .segmentationwise import SegmentationWiseFitter
from .ideal import IDEALFitter
from .segmented import SegmentedFitter

_REGISTRY: dict[str, type] = {
    "pixelwise": PixelWiseFitter,
    "segmentationwise": SegmentationWiseFitter,
    "ideal": IDEALFitter,
    "segmented": SegmentedFitter,
}


def get_fitter(name: str, **kwargs) -> BaseFitter:
    """Return a new instance of the named fitter.

    Parameters
    ----------
    name : str
        Registered fitter name. One of ``"pixelwise"``,
        ``"segmentationwise"``, ``"ideal"``.
    **kwargs
        Forwarded to the fitter constructor.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown fitter: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


__all__ = [
    "BaseFitter",
    "PixelWiseFitter",
    "SegmentationWiseFitter",
    "IDEALFitter",
    "SegmentedFitter",
    "get_fitter",
]
