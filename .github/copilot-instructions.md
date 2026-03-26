# Copilot Instructions — Pyneapple

Pyneapple is a Python library for fitting multi-exponential signal models to diffusion-weighted MRI data. It follows a **three-layer architecture** inspired by scikit-learn estimator conventions.

## Architecture

| Layer | Role | Base class | Examples |
|-------|------|------------|----------|
| **Model** | Forward physics (stateless) | `BaseModel` | `MonoExpModel`, `BiExpModel`, `TriExpModel`, `NNLSModel` |
| **Solver** | Optimization backend | `BaseSolver` | `CurveFitSolver`, `ConstrainedCurveFitSolver`, `NNLSSolver` |
| **Fitter** | Spatial orchestration | `BaseFitter` | `PixelWiseFitter`, `SegmentationWiseFitter`, `IDEALFitter` |

**Models have no `fit()`. Solvers have no spatial awareness. Fitters coordinate both.**

Data flows: TOML config → `load_config()` → `FittingConfig.build_fitter()` → Model → Solver → Fitter → `fitter.fit(xdata, image, segmentation)`.

## Key interfaces

### BaseSolver

```python
class BaseSolver(ABC):
    def __init__(self, model, max_iter=250, tol=1e-8, verbose=False, **solver_kwargs):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.diagnostics_: dict[str, Any] = {}
        self.params_: dict[str, Any] = {}

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseSolver": ...
    def get_diagnostics(self) -> dict[str, Any]: ...  # raises RuntimeError if empty
    def get_params(self) -> dict[str, Any]: ...        # raises RuntimeError if empty
    def _reset_state(self): ...                        # clears params_ and diagnostics_
```

Every `fit()` must: call `_reset_state()` first, populate `self.params_` and `self.diagnostics_`, return `self`.

### ParametricModel

```python
class ParametricModel(BaseModel):
    param_names: list[str]                    # free parameter names (excludes fixed)
    n_params: int                             # len(param_names)
    fixed_params: dict[str, float]            # model-level fixed values
    _all_param_names: list[str]               # all params including fixed

    def forward(xdata, *params) -> np.ndarray
    def jacobian(xdata, *params) -> np.ndarray | None
    def forward_with_fixed(xdata, fixed_dict, *free_params) -> np.ndarray
    def jacobian_with_fixed(xdata, fixed_dict, *free_params) -> np.ndarray | None
```

### DistributionModel

```python
class DistributionModel(BaseModel):
    bins: np.ndarray          # shape (n_bins,)
    n_bins: int
    def get_basis(xdata) -> np.ndarray   # shape (n_measurements, n_bins)
    def forward(xdata, *spectrum) -> np.ndarray  # basis @ spectrum
```

### BaseFitter

```python
class BaseFitter(ABC):
    def __init__(self, solver: BaseSolver, **fitter_kwargs):
        self.solver = solver
        self.results_: Any = None
        self.fitted_params_: dict = {}

    def fit(self, xdata, image, segmentation=None, **fit_kwargs) -> "BaseFitter": ...
    def predict(self, xdata, **predict_kwargs) -> np.ndarray: ...
    def get_fitted_params(self) -> dict[str, np.ndarray] | None: ...
```

### Registry system (`io/toml.py`)

```python
_MODEL_REGISTRY  = {"monoexp": MonoExpModel, "biexp": BiExpModel, "triexp": TriExpModel, "nnls": NNLSModel}
_SOLVER_REGISTRY = {"curvefit": CurveFitSolver, "constrained_curvefit": ConstrainedCurveFitSolver, "nnls": NNLSSolver}
_FITTER_REGISTRY = {"pixelwise": PixelWiseFitter}
```

New models/solvers/fitters should be added to the corresponding registry.

### TOML config format

```toml
[Fitting]
fitter = "pixelwise"

[Fitting.model]
type = "biexp"
fit_reduced = true
fit_s0 = false

[Fitting.solver]
type = "curvefit"
max_iter = 250
tol = 1e-8

[Fitting.solver.p0]
f1 = 0.3
D1 = 0.01
D2 = 0.001

[Fitting.solver.bounds]
f1 = [0.0, 1.0]
D1 = [0.001, 0.1]
D2 = [0.0001, 0.01]
```

## Naming conventions

- **Files:** `snake_case.py` — group by layer prefix: `test_solver_curvefit.py`, `test_fitter_pixelwise.py`
- **Classes:** `PascalCase` with layer suffix: `MonoExpModel`, `CurveFitSolver`, `PixelWiseFitter`
- **Methods:** `snake_case` — public: `fit()`, `predict()`, `get_params()`; private: `_fit_single()`
- **Fitted state:** trailing underscore (scikit-learn convention): `params_`, `diagnostics_`, `fitted_params_`

## Imports

```python
# Order: stdlib → third-party → internal (always relative)
from __future__ import annotations
from typing import Any

import numpy as np
from loguru import logger

from .base import BaseSolver
```

**Never use absolute imports** (`from pyneapple.models import ...`) within the package. Always relative.

## Type hints

- `from __future__ import annotations` in every module
- Parameter dicts: `dict[str, float | np.ndarray]`
- Bounds: `dict[str, tuple[float, float]] | None`
- Use `Any` when the layer does not need to know the concrete type
- Use `TYPE_CHECKING` for import-only dependencies

## Logging

Uses **loguru** via `from loguru import logger`.

| Level | Use for |
|-------|---------|
| `debug` | Internal state, development only |
| `info` | Progress/diagnostics, **gated behind `if self.verbose:`** |
| `warning` | Non-fatal issues where execution continues (e.g. single pixel fails in batch) |

**Never `logger.error()` then `raise`** — the exception is the error. Log once at the entry point (fitter/script), not in internal methods.

## Error handling

| Situation | Exception |
|-----------|-----------|
| Invalid input | `ValueError("descriptive message")` |
| Missing prerequisite | `RuntimeError("Call fit() first")` |
| Unsupported feature | `NotImplementedError("...")` |
| Optional dependency | `ImportError("pygpufit required for GPU fitting")` |
| Batch pixel failure | `logger.warning()` → return zeros → continue |

Internal methods just `raise`. No logging needed inside them.

## Docstrings

Google-style with numpy `Examples` block:

```python
def fit(self, xdata: np.ndarray, ydata: np.ndarray) -> "CurveFitSolver":
    """Fit model parameters to observed data.

    Args:
        xdata: Independent variable (e.g., b-values), shape (n_measurements,)
        ydata: Observed signal, shape (n_measurements,) or (n_pixels, n_measurements)

    Returns:
        CurveFitSolver: self, with fitted parameters in self.params_

    Raises:
        ValueError: If xdata and ydata shapes are incompatible

    Examples
    --------
    >>> solver.fit(bvalues, signal)
    >>> print(solver.params_)
    """
```

## Method signatures

- `Solver.fit(xdata, ydata, **fit_kwargs) -> Self` — returns `self` for chaining
- `Fitter.fit(xdata, image, segmentation, **fit_kwargs) -> Self` — returns `self`
- `Solver.predict(xdata) / Fitter.predict(xdata) -> np.ndarray` — takes measurement points, not images

## Testing

- **Framework:** pytest with `pytest-mock`, `pytest-cov`, `pytest-order`
- **File naming:** `test_<layer>_<concept>.py` (e.g., `test_solver_nnls.py`)
- **Class-based** grouping preferred: `TestNNLSSolverFitting`
- **Every test has a docstring**
- **Mocking:** always `pytest-mock` (`mocker` fixture), never `unittest.mock`
- **NumPy assertions:** `np.testing.assert_allclose(a, b, rtol=1e-5)`
- **Markers:** `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`
- Run: `uv run pytest tests/`

## Build and tooling

- **Build:** hatchling, `src/` layout
- **Python:** ≥ 3.12
- **Formatter:** black (line-length 88)
- **Linter:** ruff (line-length 88, target py312)
- **Package manager:** uv
- **CLI:** `pyneapple-pixelwise` entry point via `pyneapple.cli.pixelwise:main`

## Project layout

```
src/pyneapple/
├── cli/              # CLI entry points (argparse)
├── fitters/          # Spatial orchestrators (BaseFitter, PixelWise, Segmentation, IDEAL)
├── io/               # I/O + registry (bvalue, hdf5, nifti, toml)
├── model_functions/  # Pure forward functions (multiexp.py, nnls.py)
├── models/           # Model classes (base.py, monoexp, biexp, triexp, nnls)
├── solvers/          # Optimization backends (base, curvefit, constrained, nnls)
└── utility/          # Input validation
```
