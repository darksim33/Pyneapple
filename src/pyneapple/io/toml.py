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

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as _entry_points_raw
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import numpy as np
from loguru import logger


def entry_points(group: str):
    """Compatibility shim: ``importlib.metadata.entry_points(group=)`` was
    added in Python 3.12.  On older runtimes the function returns a plain
    dict, so we fall back to ``dict.get(group, [])``.
    """
    if sys.version_info >= (3, 12):
        return _entry_points_raw(group=group)
    return _entry_points_raw().get(group, [])


from ..fitters import (
    IDEALFitter,
    PixelWiseFitter,
    SegmentationWiseFitter,
    SegmentedFitter,
)
from ..fitters.base import BaseFitter
from ..models import BiExpModel, MonoExpModel, NNLSModel, TriExpModel
from ..models.base import DistributionModel
from ..solvers import ConstrainedCurveFitSolver, CurveFitSolver, NNLSSolver

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
# Internal builder helpers (reused for both top-level and step-1 configs)
# ---------------------------------------------------------------------------


def _build_model(
    model_type: str,
    model_kwargs: dict[str, Any],
    fixed_params: dict[str, float],
):
    """Instantiate and return a model from registry name and kwargs.

    Args:
        model_type: Key in ``_MODEL_REGISTRY``.
        model_kwargs: Extra keyword arguments for the model constructor.
        fixed_params: Fixed parameter values to attach to the model.  Only
            applied to non-distribution models.

    Returns:
        Configured model instance.
    """
    model_cls = _resolve(_MODEL_REGISTRY, model_type)
    kwargs: dict[str, Any] = {**model_kwargs}
    if "d_range" in kwargs:
        kwargs["d_range"] = tuple(kwargs["d_range"])
    if not issubclass(model_cls, DistributionModel) and fixed_params:
        kwargs["fixed_params"] = fixed_params
    model = model_cls(**kwargs)
    param_info = getattr(model, "param_names", "distribution")
    logger.info(f"Built model: {model_cls.__name__} | params={param_info}")
    return model


def _build_solver(
    solver_type: str,
    model,
    solver_kwargs: dict[str, Any],
    p0: dict[str, float],
    bounds: dict[str, tuple[float, float]],
):
    """Instantiate and return a solver from registry name and configuration.

    Args:
        solver_type: Key in ``_SOLVER_REGISTRY``.
        model: Model instance to attach to the solver.
        solver_kwargs: Keyword arguments for the solver constructor (must
            include ``max_iter`` and ``tol``).
        p0: Per-parameter initial guesses.
        bounds: Per-parameter ``(lo, hi)`` bounds.

    Returns:
        Configured solver instance.
    """
    solver_cls = _resolve(_SOLVER_REGISTRY, solver_type)
    if isinstance(model, DistributionModel):
        solver = solver_cls(model=model, **solver_kwargs)
    else:
        solver = solver_cls(
            model=model,
            p0=p0,
            bounds=bounds,
            **solver_kwargs,
        )
    logger.info(
        f"Built solver: {solver_cls.__name__} | "
        f"max_iter={solver.max_iter}, tol={solver.tol}"
    )
    return solver


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
    segmented_kwargs: dict[str, Any] = field(default_factory=dict)

    def build_fitter(self) -> BaseFitter:
        """Instantiate and return a fully configured fitter.

        Builds the model → solver → fitter in order, logging each step.

        Returns:
            BaseFitter: Ready-to-call fitter instance.

        Raises:
            KeyError: If any registered type is not found in its registry.
            ValueError: If required config sections are missing.
        """
        fitter_cls = _resolve(_FITTER_REGISTRY, self.fitter_type)

        if self.fitter_type == "segmented":
            return self._build_segmented_fitter(fitter_cls)

        # --- step-2 / only model+solver for all non-segmented fitters ---
        model = _build_model(self.model_type, self.model_kwargs, self.fixed_params)
        solver = _build_solver(
            self.solver_type, model, self.solver_kwargs, self.p0, self.bounds
        )

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

    def _build_segmented_fitter(self, fitter_cls) -> BaseFitter:
        """Build a :class:`SegmentedFitter` from *segmented_kwargs*.

        Requires the ``[Fitting.segmented]`` section to have been parsed into
        :attr:`segmented_kwargs` by :func:`load_config`.

        Args:
            fitter_cls: The ``SegmentedFitter`` class (already resolved).

        Returns:
            BaseFitter: Fully configured ``SegmentedFitter`` instance.

        Raises:
            ValueError: If ``segmented_kwargs`` is empty (section missing in
                the TOML file).
        """
        if not self.segmented_kwargs:
            raise ValueError(
                "fitter = 'segmented' requires a [Fitting.segmented] section "
                "in the config file."
            )
        sk = self.segmented_kwargs

        # Step 2 uses the top-level [Fitting.model] / [Fitting.solver]
        step2_model = _build_model(
            self.model_type, self.model_kwargs, self.fixed_params
        )
        step2_solver = _build_solver(
            self.solver_type, step2_model, self.solver_kwargs, self.p0, self.bounds
        )

        # Step 1 uses the parsed [Fitting.segmented.step1.*] data
        step1_model = _build_model(sk["step1_model_type"], sk["step1_model_kwargs"], {})
        step1_solver = _build_solver(
            sk["step1_solver_type"],
            step1_model,
            sk["step1_solver_kwargs"],
            sk["step1_p0"],
            sk["step1_bounds"],
        )

        fitter = fitter_cls(
            step1_solver=step1_solver,
            step2_solver=step2_solver,
            step1_bvalue_range=sk.get("step1_bvalue_range"),
            fixed_from_step1=sk.get("fixed_from_step1"),
            param_mapping=sk.get("param_mapping"),
        )
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
    model_kwargs = {
        k.lower(): v for k, v in model_cfg.items() if k.lower() in _MODEL_KWARG_KEYS
    }

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

    # SegmentedFitter kwargs from optional [Fitting.segmented] section
    segmented_kwargs: dict[str, Any] = {}
    if fitter_type == "segmented":
        seg_raw: dict[str, Any] = dict(fitting.get("segmented", {}))
        if not seg_raw:
            raise ValueError(
                "fitter = 'segmented' requires a [Fitting.segmented] section in "
                "the config file."
            )

        # --- Step 1 model ---
        step1_raw: dict[str, Any] = dict(seg_raw.get("step1", {}))
        step1_model_cfg: dict[str, Any] = dict(step1_raw.get("model", {}))
        step1_model_type = str(step1_model_cfg.pop("type", "monoexp")).lower()
        if step1_model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown step1 model type: {step1_model_type!r}. "
                f"Available: {sorted(_MODEL_REGISTRY)}"
            )
        step1_model_kwargs = {
            k: v for k, v in step1_model_cfg.items() if k in _MODEL_KWARG_KEYS
        }

        # --- Step 1 solver ---
        step1_solver_cfg: dict[str, Any] = dict(step1_raw.get("solver", {}))
        step1_solver_type = str(step1_solver_cfg.get("type", "curvefit")).lower()
        if step1_solver_type not in _SOLVER_REGISTRY:
            raise ValueError(
                f"Unknown step1 solver type: {step1_solver_type!r}. "
                f"Available: {sorted(_SOLVER_REGISTRY)}"
            )
        step1_max_iter = int(step1_solver_cfg.get("max_iter", 250))
        step1_tol = float(step1_solver_cfg.get("tol", 1e-8))
        step1_p0 = {k: float(v) for k, v in step1_solver_cfg.get("p0", {}).items()}
        step1_bounds_raw: dict[str, Any] = step1_solver_cfg.get("bounds", {})
        step1_bounds: dict[str, tuple[float, float]] = {}
        for param, rng in step1_bounds_raw.items():
            if len(rng) != 2:
                raise ValueError(
                    f"Step1 bounds for '{param}' must be a two-element list "
                    f"[lo, hi], got: {rng}"
                )
            step1_bounds[param] = (float(rng[0]), float(rng[1]))
        step1_extra_kwargs = {
            k: v
            for k, v in step1_solver_cfg.items()
            if k not in _SOLVER_RESERVED_KEYS and not isinstance(v, dict)
        }
        step1_solver_kwargs: dict[str, Any] = {
            "max_iter": step1_max_iter,
            "tol": step1_tol,
            **step1_extra_kwargs,
        }

        # --- step1_bvalue_range: [200, null] → (200.0, None) ---
        brange_raw = seg_raw.get("step1_bvalue_range")
        if brange_raw is not None:
            lo = float(brange_raw[0]) if brange_raw[0] is not None else None
            hi = float(brange_raw[1]) if brange_raw[1] is not None else None
            step1_bvalue_range: tuple[float | None, float | None] | None = (lo, hi)
        else:
            step1_bvalue_range = None

        segmented_kwargs = {
            "step1_bvalue_range": step1_bvalue_range,
            "fixed_from_step1": list(seg_raw.get("fixed_from_step1", [])),
            "param_mapping": dict(seg_raw.get("param_mapping", {})),
            "step1_model_type": step1_model_type,
            "step1_model_kwargs": step1_model_kwargs,
            "step1_solver_type": step1_solver_type,
            "step1_solver_kwargs": step1_solver_kwargs,
            "step1_p0": step1_p0,
            "step1_bounds": step1_bounds,
        }

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
        segmented_kwargs=segmented_kwargs,
    )

    logger.info(
        f"Loaded config from '{path}': "
        f"fitter={fitter_type}, model={model_type}, solver={solver_type}"
    )
    return config
