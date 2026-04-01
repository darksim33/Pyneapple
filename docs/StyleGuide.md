# Pyneapple Style Guide

Coding conventions for the Pyneapple codebase.

## Architecture Layers

| Layer    | Role                          | Example Classes                      |
|----------|-------------------------------|--------------------------------------|
| Model    | Forward physics (stateless)   | `MonoExpModel`, `BiexponentialModel`  |
| Solver   | Optimization backend          | `CurveFitSolver`, `NNLSSolver`       |
| Fitter   | Spatial orchestration         | `BaseFitter`, `PixelWiseFitter`      |

Models have no `fit()`. Solvers have no spatial awareness. Fitters coordinate both.

## Naming

- **Files:** `snake_case.py` — group by concept prefix: `test_solver_curvefit.py`, `test_solver_nnls.py`
- **Classes:** `PascalCase` — suffix by layer: `MonoExpModel`, `CurveFitSolver`, `PixelWiseFitter`
- **Methods:** `snake_case` — public interface: `fit()`, `predict()`, `get_params()`
- **Private:** single underscore prefix: `_fit_single()`, `_prepare_bounds()`
- **Fitted state:** trailing underscore (scikit-learn convention): `params_`, `diagnostics_`

## Logging

Uses **loguru** via `from loguru import logger`.
Change logging level by: 
```python
import pyneapple

# Default: WARNING level
# Configure for different verbosity
pyneapple.configure_logging(level='INFO')  # More verbose
pyneapple.configure_logging(level='DEBUG')  # Maximum verbosity
pyneapple.configure_logging(level='ERROR')  # Minimal output
```

### Levels

| Level   | Use for                                            | Example                                  |
|---------|----------------------------------------------------|------------------------------------------|
| `debug` | Internal state, only useful for development        | Validation details                        |
| `info`  | Progress/diagnostics, gated behind `self.verbose`  | `"Fitting 1024 pixels sequentially..."`   |
| `warning` | Non-fatal issues where execution continues       | `"Fit failed for pixel 42: {e}"`          |
| `error` | **Never log always raise** (see below)             |                                         |

1. **Never `logger.error()` then `raise`** — the exception *is* the error. The redundant log creates duplicate noise, especially with file sinks.
2. **Use `logger.warning()`** for caught exceptions where execution continues (e.g., single pixel fails in a batch, return zeros and keep going).
3. **Gate `logger.info()` behind `if self.verbose:`** — callers opt in to progress output.
4. **Log errors at the boundary** — the top-level entry point (Fitter, script) wraps calls in `try/except` and logs once:
   ```python
   # In Fitter or script — single log entry per error
   try:
       solver.fit(xdata, ydata, **fit_kwargs)
   except Exception as e:
       logger.error(f"Fitting failed for segment {seg_id}: {e}")
       raise
   ```
5. **Internal methods just `raise`** with clear messages. No logging needed.

## Error Handling

- **Validation errors:** `raise ValueError("descriptive message")`
- **Missing prerequisites:** `raise RuntimeError("Call fit() first")`
- **Unsupported features:** `raise NotImplementedError("Reg order 4 not supported")`
- **Optional dependencies:** `raise ImportError("pygpufit is required for GPU fitting")`
- **Batch pixel failures:** catch, `logger.warning()`, return zeros, continue

## Type Hints

- Use `from __future__ import annotations` in every module.
- Parameter dicts: `dict[str, float | np.ndarray]`
- Bounds: `dict[str, tuple[float, float]] | None`
- Use `Any` for model parameters when the layer doesn't need to know the concrete type.
- Use `TYPE_CHECKING` for imports only needed by type checkers.

## Docstrings

Google-style with numpy-style `Examples` block:

```python
def fit(self, xdata: np.ndarray, ydata: np.ndarray) -> "SolverClass":
    """Fit model parameters to observed data.

    Args:
        xdata: Independent variable (e.g., b-values), shape (n_measurements,)
        ydata: Observed signal, shape (n_measurements,) or (..., n_measurements)

    Returns:
        SolverClass: self, with fitted parameters in self.params_

    Raises:
        ValueError: If xdata and ydata shapes are incompatible

    Examples
    --------
    >>> solver.fit(bvalues, signal)
    >>> print(solver.params_)
    """
```

## Method Signatures

### Solver.fit()
```python
def fit(self, xdata, ydata, **fit_kwargs) -> Self
```
- Returns `self` for optional chaining

### Fitter.fit()
```python
def fit(self, image, segmentation, xdata) -> Self
```

### Fitter.predict() / Solver.predict()
```python
def predict(self, xdata) -> np.ndarray
```
- Takes measurement points (b-values), **not** image data
- Returns reconstructed signal at those points

## Tests

See [TestingGuidelines.md](TestingGuidelines.md) for comprehensive testing conventions.

Key points:

- **Framework:** pytest
- **File naming:** `test_<layer>_<concept>.py` (e.g., `test_solver_nnls.py`, `test_solver_curvefit.py`)
- **Class grouping:** group related tests in classes: `TestNNLSSolverBasics`, `TestNNLSSolverFitting`
- **Markers:** `@pytest.mark.slow` for expensive tests, `@pytest.mark.integration` for end-to-end
- **Fixtures:** shared fixtures in `conftest.py` or at module level
- **Tolerances:** `np.testing.assert_allclose()` with explicit `rtol`/`atol`
- **Timing:** use `time.perf_counter()` with `capsys.disabled()` for direct output

## Imports

```python
# Standard library
from __future__ import annotations
from typing import Any

# Third-party
import numpy as np

# Internal — always relative within pyneapple
from loguru import logger
from .base import BaseSolver
```

Never use absolute imports (`from pyneapple.utils import ...`) within the package.
