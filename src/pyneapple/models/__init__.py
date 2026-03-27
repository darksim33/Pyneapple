from .base import BaseModel, ParametricModel, DistributionModel
from .monoexp import MonoExpModel
from .biexp import BiExpModel
from .triexp import TriExpModel
from .nnls import NNLSModel

_REGISTRY: dict[str, type] = {
    "monoexp": MonoExpModel,
    "biexp": BiExpModel,
    "triexp": TriExpModel,
    "nnls": NNLSModel,
}


def get_model(name: str) -> BaseModel:
    """Return a new instance of the named model.

    Parameters
    ----------
    name : str
        Registered model name. One of ``"monoexp"``, ``"biexp"``,
        ``"triexp"``, ``"nnls"``.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


__all__ = [
    "BaseModel",
    "ParametricModel",
    "DistributionModel",
    "MonoExpModel",
    "BiExpModel",
    "TriExpModel",
    "NNLSModel",
    "get_model",
]
