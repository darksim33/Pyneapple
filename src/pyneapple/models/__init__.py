from __future__ import annotations

import inspect

from .base import BaseModel, DistributionModel, ParametricModel
from .biexp import BiExpModel
from .monoexp import MonoExpModel
from .nnls import NNLSModel
from .triexp import TriExpModel

_REGISTRY: dict[str, type] = {
    "monoexp": MonoExpModel,
    "biexp": BiExpModel,
    "triexp": TriExpModel,
    "nnls": NNLSModel,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Return a new instance of the named model.

    Parameters
    ----------
    name : str
        Registered model name. One of ``"monoexp"``, ``"biexp"``,
        ``"triexp"``, ``"nnls"``.
    **kwargs
        Constructor keyword arguments forwarded to the model class.
        Required for models whose ``__init__`` has no defaults (e.g.
        ``NNLSModel`` requires ``d_range`` and ``n_bins``).

    Raises
    ------
    ValueError
        If *name* is not in the registry, or if required constructor
        arguments are missing from *kwargs*.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {sorted(_REGISTRY)}")
    model_cls = _REGISTRY[key]
    sig = inspect.signature(model_cls.__init__)
    missing = [
        p.name
        for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and p.name not in kwargs
    ]
    if missing:
        raise ValueError(
            f"get_model({name!r}) is missing required argument(s): {missing}. "
            f"Pass them as keyword arguments, e.g. get_model({name!r}, {missing[0]}=...)."
        )
    return model_cls(**kwargs)


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
