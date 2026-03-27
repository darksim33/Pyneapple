"""TOML configuration file reader for Pyneapple fitting pipelines.

Parses a structured TOML config describing a model, solver, and fitter and
returns a :class:`FittingConfig` that can instantiate a ready-to-use fitter.

Expected file layout::

    [Fitting]
    fitter = "pixelwise"

    [Fitting.model]
    type = "triexp"
    fit_reduced = false
    fit_s0 = false
    fit_t1 = false
    fit_t1_steam = false

    [Fitting.solver]
    type = "curvefit"
    max_iter = 250
    tol = 1e-8

    [Fitting.solver.p0]
    f1 = 85.0

    [Fitting.solver.bounds]
    f1 = [10.0, 500.0]
"""

import tomllib
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from loguru import logger

import numpy as np

from ..models import MonoExpModel, BiExpModel, TriExpModel, NNLSModel
from ..models.base import DistributionModel
from ..solvers import CurveFitSolver, ConstrainedCurveFitSolver, NNLSSolver
from ..fitters import (
    PixelWiseFitter,
    SegmentationWiseFitter,
    IDEALFitter,
    SegmentedFitter,
)
from ..fitters.base import BaseFitter


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type] = {
    "monoexp": MonoExpModel,
    "biexp": BiExpModel,
    "triexp": TriExpModel,
    "nnls": NNLSModel,
}

_SOLVER_REGISTRY: dict[str, type] = {
    "curvefit": CurveFitSolver,
    "constrained_curvefit": ConstrainedCurveFitSolver,
    "nnls": NNLSSolver,
}

_FITTER_REGISTRY: dict[str, type] = {
    "pixelwise": PixelWiseFitter,
    "segmentationwise": SegmentationWiseFitter,
    "ideal": IDEALFitter,
    "segmented": SegmentedFitter,
}

# Keys in [Fitting.model] that are forwarded as kwargs to the model constructor.
_MODEL_KWARG_KEYS: frozenset[str] = frozenset(
    {
        "fit_reduced",
        "fit_s0",
        "fit_t1",
        "fit_t1_steam",
        "repetition_time",
        "mixing_time",
        "d_range",
        "n_bins",
    }
)

# Keys to skip when collecting extra solver kwargs (handled explicitly).
_SOLVER_RESERVED_KEYS: frozenset[str] = frozenset(
    {"type", "max_iter", "tol", "p0", "bounds", "fraction_constraint"}
)


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


def _discover_plugins(group: str, registry: dict[str, type | EntryPoint]) -> None:
    """Merge installed entry-point plugins into a registry.

    Reads only package metadata — the plugin module is **not** imported at this
    point.  The raw :class:`~importlib.metadata.EntryPoint` object is stored
    instead and loaded lazily on first use via :func:`_resolve`.

    Args:
        group: Entry-point group name (e.g. ``"pyneapple.solvers"``).
        registry: Mutable registry dict to update in-place. Built-in entries
            are never overwritten.
    """
    for ep in entry_points(group=group):
        if ep.name not in registry:
            registry[ep.name] = ep
            logger.debug(f"Discovered plugin: [{group}] {ep.name} → {ep.value}")


def _resolve(registry: dict[str, type | EntryPoint], key: str) -> type:
    """Return the class for *key*, loading an EntryPoint on first access.

    If the registry value is a plain class it is returned immediately.  If it
    is an :class:`~importlib.metadata.EntryPoint` the module is imported,
    the loaded class replaces the entry-point in the registry (so subsequent
    calls skip the import), and the class is returned.

    Args:
        registry: One of the module-level ``_*_REGISTRY`` dicts.
        key: The type name as it appears in the TOML config.

    Returns:
        type: The resolved class.
    """
    cls = registry[key]
    if isinstance(cls, EntryPoint):
        cls = cls.load()
        registry[key] = cls  # cache — imported once, reused thereafter
    return cls


_discover_plugins("pyneapple.solvers", _SOLVER_REGISTRY)
_discover_plugins("pyneapple.models", _MODEL_REGISTRY)
_discover_plugins("pyneapple.fitters", _FITTER_REGISTRY)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class FittingConfig:
    """Parsed and validated fitting configuration.

    Attributes:
        fitter_type: Registered fitter name (e.g. ``"pixelwise"``).
        model_type: Registered model name (e.g. ``"triexp"``).
        solver_type: Registered solver name (e.g. ``"curvefit"``).
        model_kwargs: Extra keyword arguments forwarded to the model constructor.
        solver_kwargs: Extra keyword arguments forwarded to the solver constructor
            (excludes ``p0`` and ``bounds`` which are passed separately).
        p0: Per-parameter initial guesses.
        bounds: Per-parameter (lower, upper) bounds.
    """

    fitter_type: str
    model_type: str
    solver_type: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    p0: dict[str, float] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    fixed_params: dict[str, float] = field(default_factory=dict)
    ideal_kwargs: dict[str, Any] = field(default_factory=dict)

    def build_fitter(self) -> BaseFitter:
        """Instantiate and return a fully configured fitter.

        Builds the model → solver → fitter in order, logging each step.

        Returns:
            BaseFitter: Ready-to-call fitter instance.

        Raises:
            KeyError: If any registered type is not found in its registry.
        """
        model_cls = _resolve(_MODEL_REGISTRY, self.model_type)
        model_kwargs = {**self.model_kwargs}
        if "d_range" in model_kwargs:
            model_kwargs["d_range"] = tuple(model_kwargs["d_range"])
        if not issubclass(model_cls, DistributionModel) and self.fixed_params:
            model_kwargs["fixed_params"] = self.fixed_params
        model = model_cls(**model_kwargs)
        param_info = getattr(model, "param_names", "distribution")
        logger.info(f"Built model: {model_cls.__name__} | params={param_info}")

        solver_cls = _resolve(_SOLVER_REGISTRY, self.solver_type)
        if isinstance(model, DistributionModel):
            solver = solver_cls(model=model, **self.solver_kwargs)
        else:
            solver = solver_cls(
                model=model,
                p0=self.p0,
                bounds=self.bounds,
                **self.solver_kwargs,
            )
        logger.info(
            f"Built solver: {solver_cls.__name__} | "
            f"max_iter={solver.max_iter}, tol={solver.tol}"
        )

        fitter_cls = _resolve(_FITTER_REGISTRY, self.fitter_type)
        if self.fitter_type == "ideal":
            if not self.ideal_kwargs:
                raise ValueError(
                    "IDEAL fitter requires a [Fitting.ideal] section in the config "
                    "with at least 'dim_steps' and 'step_tol'."
                )
            ideal_kw = dict(self.ideal_kwargs)
            dim_steps = np.array(ideal_kw.pop("dim_steps"))
            step_tol = ideal_kw.pop("step_tol")
            fitter = fitter_cls(
                solver=solver,
                dim_steps=dim_steps,
                step_tol=step_tol,
                **ideal_kw,
            )
        else:
            fitter = fitter_cls(solver=solver)
        logger.info(f"Built fitter: {fitter_cls.__name__}")

        return fitter


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> FittingConfig:
    """Load and parse a Pyneapple TOML configuration file.

    Args:
        path: Path to the ``.toml`` configuration file.

    Returns:
        FittingConfig: Parsed configuration object. Call
            :meth:`FittingConfig.build_fitter` to obtain a ready-to-use fitter.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If ``[Fitting]`` section is missing.
        ValueError: If an unknown fitter, model, or solver type is specified,
            or if bounds entries are not two-element lists.
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Please check the file path and ensure the file exists."
        )

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    if "Fitting" not in raw:
        raise KeyError(
            "Missing required [Fitting] section in config file. "
            f"Found top-level keys: {list(raw.keys())}"
        )

    fitting = raw["Fitting"]

    # --- Fitter ---
    fitter_type = str(fitting.get("fitter", "pixelwise")).lower()
    if fitter_type not in _FITTER_REGISTRY:
        raise ValueError(
            f"Unknown fitter type: {fitter_type!r}. "
            f"Available: {sorted(_FITTER_REGISTRY)}"
        )

    # --- Model ---
    model_cfg: dict[str, Any] = dict(fitting.get("model", {}))
    model_type = str(model_cfg.pop("type", "monoexp")).lower()
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Available: {sorted(_MODEL_REGISTRY)}"
        )
    model_kwargs = {k: v for k, v in model_cfg.items() if k in _MODEL_KWARG_KEYS}

    # fixed_params: {param: float} — parameters held constant during fitting
    fixed_params_raw: dict[str, Any] = model_cfg.get("fixed_params", {})
    fixed_params: dict[str, float] = {k: float(v) for k, v in fixed_params_raw.items()}

    # --- Solver ---
    solver_cfg: dict[str, Any] = dict(fitting.get("solver", {}))
    solver_type = str(solver_cfg.get("type", "curvefit")).lower()
    if solver_type not in _SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver type: {solver_type!r}. "
            f"Available: {sorted(_SOLVER_REGISTRY)}"
        )

    max_iter: int = int(solver_cfg.get("max_iter", 250))
    tol: float = float(solver_cfg.get("tol", 1e-8))

    # p0: {param: float}
    p0_raw: dict[str, Any] = solver_cfg.get("p0", {})
    p0 = {k: float(v) for k, v in p0_raw.items()}

    # bounds: {param: (lo, hi)} — config stores as two-element list
    bounds_raw: dict[str, Any] = solver_cfg.get("bounds", {})
    bounds: dict[str, tuple[float, float]] = {}
    for param, rng in bounds_raw.items():
        if len(rng) != 2:
            raise ValueError(
                f"Bounds for '{param}' must be a two-element list [lo, hi], got: {rng}"
            )
        bounds[param] = (float(rng[0]), float(rng[1]))

    # Any remaining scalar solver_cfg entries are forwarded as extra kwargs.
    extra_solver_kwargs = {
        k: v
        for k, v in solver_cfg.items()
        if k not in _SOLVER_RESERVED_KEYS and not isinstance(v, dict)
    }

    solver_kwargs: dict[str, Any] = {
        "max_iter": max_iter,
        "tol": tol,
        **extra_solver_kwargs,
    }

    # fraction_constraint is a reserved key specific to constrained_curvefit
    if "fraction_constraint" in solver_cfg:
        solver_kwargs["fraction_constraint"] = bool(solver_cfg["fraction_constraint"])

    # IDEAL fitter kwargs from optional [Fitting.ideal] section
    ideal_kwargs: dict[str, Any] = {}
    if fitter_type == "ideal":
        ideal_raw: dict[str, Any] = dict(fitting.get("ideal", {}))
        if not ideal_raw:
            raise ValueError(
                "fitter = 'ideal' requires a [Fitting.ideal] section with at least "
                "'dim_steps' and 'step_tol'."
            )
        if "dim_steps" not in ideal_raw:
            raise ValueError(
                "Missing required key 'dim_steps' in [Fitting.ideal] section."
            )
        if "step_tol" not in ideal_raw:
            raise ValueError(
                "Missing required key 'step_tol' in [Fitting.ideal] section."
            )
        ideal_kwargs = ideal_raw

    config = FittingConfig(
        fitter_type=fitter_type,
        model_type=model_type,
        solver_type=solver_type,
        model_kwargs=model_kwargs,
        solver_kwargs=solver_kwargs,
        p0=p0,
        bounds=bounds,
        fixed_params=fixed_params,
        ideal_kwargs=ideal_kwargs,
    )

    logger.info(
        f"Loaded config from '{path}': "
        f"fitter={fitter_type}, model={model_type}, solver={solver_type}"
    )
    return config
