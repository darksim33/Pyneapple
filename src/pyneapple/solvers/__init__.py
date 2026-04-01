from __future__ import annotations

from .base import BaseSolver
from .curvefit import CurveFitSolver
from .constrained_curvefit import ConstrainedCurveFitSolver
from .nnls_solver import NNLSSolver

_REGISTRY: dict[str, type] = {
    "curvefit": CurveFitSolver,
    "constrained_curvefit": ConstrainedCurveFitSolver,
    "nnls": NNLSSolver,
}


def get_solver(name: str, **kwargs) -> BaseSolver:
    """Return a new instance of the named solver.

    Parameters
    ----------
    name : str
        Registered solver name. One of ``"curvefit"``,
        ``"constrained_curvefit"``, ``"nnls"``.
    **kwargs
        Forwarded to the solver constructor.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown solver: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


__all__ = [
    "BaseSolver",
    "CurveFitSolver",
    "ConstrainedCurveFitSolver",
    "NNLSSolver",
    "get_solver",
]
