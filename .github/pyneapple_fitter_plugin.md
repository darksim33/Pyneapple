# Copilot Instructions — Pyneapple Fitter Plugin Template

This is a **Pyneapple plugin** that provides fitters for diffusion MRI analysis.
It registers into Pyneapple's entry_point-based plugin system.

## What this package does

- Provides one or more fitters (spatial fitting strategies / orchestration pipelines)
- Each fitter inherits from `pyneapple.fitters.base.BaseFitter`
- Registers via `[project.entry-points."pyneapple.fitters"]` in `pyproject.toml`
- Heavy dependencies (CUDA, proprietary libs, etc.) are only imported when a fitter is instantiated, never at discovery time

## Pyneapple fitter contract

Every fitter **must**:

1. Inherit from `pyneapple.fitters.base.BaseFitter`
2. Call `super().__init__(solver=solver, verbose=verbose, **fitter_kwargs)`
3. Implement `fit(self, xdata, image, segmentation=None, **fit_kwargs) -> self`
4. Implement `predict(self, xdata, **predict_kwargs) -> np.ndarray`
5. Inside `fit()`:
   - Validate inputs using `pyneapple.utility.validation` helpers
   - Set `self.n_measurements`, `self.image_shape`
   - Extract pixel data via `self._extract_pixel_data(image, segmentation)`
   - Measure wall-clock time around `self.solver.fit(...)` using `time.perf_counter()`
   - Delegate optimization to `self.solver.fit(...)`
   - Store fitted parameters in `self.fitted_params_` (a `dict[str, np.ndarray]`)
   - Assemble the public result via `self.results_ = self._assemble_fit_result(xdata, pixel_signals, fit_time)`
   - Return `self`

### BaseFitter interface

```python
import time
from pyneapple.fitters.base import BaseFitter
from pyneapple.result import FitResult

class BaseFitter(ABC):
    def __init__(self, solver: BaseSolver, verbose: bool = False, **fitter_kwargs):
        self.solver = solver
        self.verbose = verbose
        self.fitter_kwargs = fitter_kwargs
        self.results_: FitResult | None = None   # populated by fit()
        self.fitted_params_: dict = {}
        self.image_shape: tuple | None = None
        self.pixel_indices: list[tuple] | None = None
        self.n_measurements: int | None = None

    @abstractmethod
    def fit(self, xdata, image, segmentation=None, **fit_kwargs) -> "BaseFitter": ...

    @abstractmethod
    def predict(self, xdata, **predict_kwargs) -> np.ndarray: ...

    def get_fitted_params(self) -> dict[str, np.ndarray] | None: ...
    def _extract_pixel_data(self, image, segmentation) -> np.ndarray: ...  # sets self.pixel_indices
    def _reconstruct_volume(self, flat_values, pixel_indices, spatial_shape) -> np.ndarray: ...
    def _check_fitted(self) -> None: ...         # raises RuntimeError if not fitted
    def _get_param_names(self) -> list[str]: ... # delegates to self.solver.model.param_names
    def _compute_r_squared(self, xdata, pixel_signals) -> np.ndarray: ...  # per-pixel R²
    def _assemble_fit_result(self, xdata, pixel_signals, fit_time,
                             pixel_indices=None) -> FitResult: ...
```

### FitResult — the public result object

`_assemble_fit_result()` reads `self.solver.pixel_results_` and builds a `FitResult`:

```python
from pyneapple.result import FitResult

# Available fields after fit():
result: FitResult = fitter.results_

result.params          # dict[str, np.ndarray] — per-pixel parameter maps
result.success         # np.ndarray(bool, shape=(n_pixels,))
result.n_iterations    # np.ndarray(int) or None (None for CurveFitSolver)
result.messages        # list[str | None] or None (populated for ConstrainedCurveFitSolver)
result.covariance      # np.ndarray(float, shape=(n_pixels, n_params, n_params)) or None
result.residuals       # np.ndarray(float, shape=(n_pixels,)) or None (NNLSSolver only)
result.r_squared       # np.ndarray(float, shape=(n_pixels,)) — per-pixel R²
result.fit_time        # float — total wall-clock seconds
result.image_shape     # tuple — original (X, Y, Z, N) image shape
result.pixel_indices   # list[tuple] — spatial (x, y, z) index per pixel
result.n_pixels        # int
result.solver_name     # str — e.g. "CurveFitSolver"
result.model_name      # str — e.g. "MonoExpModel"

# Convenience properties:
result.n_converged     # int — number of pixels where success=True
result.convergence_rate  # float [0.0, 1.0]
result.mean_r_squared  # float | None — nanmean of r_squared
```

### Pixel-wise fitter pattern

For fitting each pixel independently (parametric or distribution models):

```python
import time
import numpy as np
from pyneapple.fitters.base import BaseFitter

class MyPixelFitter(BaseFitter):
    def __init__(self, solver: BaseSolver, verbose=False, **fitter_kwargs):
        super().__init__(solver=solver, verbose=verbose, **fitter_kwargs)

    def fit(self, xdata, image, segmentation=None,
            fixed_param_maps=None, **fit_kwargs) -> "MyPixelFitter":
        # Input validation
        validate_xdata(xdata)
        validate_data_shapes(xdata, image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape

        if segmentation is not None:
            segmentation = validate_segmentation(segmentation, image.shape)
        else:
            segmentation = np.ones(image.shape[:3], dtype=int)

        pixel_signals = self._extract_pixel_data(image, segmentation)

        # Delegate to solver — measure wall-clock time
        t0 = time.perf_counter()
        self.solver.fit(xdata, pixel_signals, **fit_kwargs)
        fit_time = time.perf_counter() - t0

        # Store backwards-compatible fitted_params_
        for param, values in self.solver.params_.items():
            self.fitted_params_[param] = values

        # Assemble the public FitResult (includes R², timing, convergence)
        self.results_ = self._assemble_fit_result(xdata, pixel_signals, fit_time)

        return self

    def predict(self, xdata, **predict_kwargs) -> np.ndarray:
        return super().predict(xdata, **predict_kwargs)
```

### Segmentation-wise fitter pattern

For fitting mean signal per segmented region:

```python
import time
import numpy as np
from pyneapple.fitters.base import BaseFitter

class MySegmentFitter(BaseFitter):
    def __init__(self, solver: BaseSolver, verbose=False, **fitter_kwargs):
        super().__init__(solver=solver, verbose=verbose, **fitter_kwargs)
        self.segment_labels: np.ndarray | None = None

    def fit(self, xdata, image, segmentation=None, **fit_kwargs) -> "MySegmentFitter":
        if segmentation is None:
            raise ValueError("segmentation is required")
        validate_xdata(xdata)
        validate_data_shapes(xdata, image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape
        segmentation = validate_segmentation(segmentation, image.shape)

        # Compute mean signal per segment
        pixel_signals = self._extract_pixel_data(image, segmentation)

        t0 = time.perf_counter()
        self.solver.fit(xdata, pixel_signals, **fit_kwargs)
        fit_time = time.perf_counter() - t0

        for param, values in self.solver.params_.items():
            self.fitted_params_[param] = values

        self.results_ = self._assemble_fit_result(xdata, pixel_signals, fit_time)
        return self

    def predict(self, xdata, **predict_kwargs) -> np.ndarray:
        # Override to broadcast segment-level predictions back to pixel level
        ...
```

### Multi-step pipeline fitter pattern

For orchestrating multiple solvers in sequence (e.g., simple model then complex model):

```python
import time
import dataclasses
import numpy as np
from pyneapple.fitters.base import BaseFitter

class MyPipelineFitter(BaseFitter):
    def __init__(self, step1_solver: BaseSolver, step2_solver: BaseSolver,
                 fixed_from_step1: list[str] | None = None,
                 param_mapping: dict[str, str] | None = None,
                 verbose=False, **fitter_kwargs):
        super().__init__(solver=step2_solver, verbose=verbose, **fitter_kwargs)
        self.step1_solver = step1_solver
        self.step2_solver = step2_solver
        self.fixed_from_step1 = fixed_from_step1 or []
        self.param_mapping = param_mapping or {}
        self.step1_result_: FitResult | None = None   # intermediate result from step 1

    def fit(self, xdata, image, segmentation=None, **fit_kwargs) -> "MyPipelineFitter":
        validate_xdata(xdata)
        validate_data_shapes(xdata, image)
        self.n_measurements = len(xdata)
        self.image_shape = image.shape
        if segmentation is None:
            segmentation = np.ones(image.shape[:3], dtype=int)
        else:
            segmentation = validate_segmentation(segmentation, image.shape)

        pixel_signals = self._extract_pixel_data(image, segmentation)

        # Start overall wall clock before step 1
        t_total = time.perf_counter()

        # Step 1: fit simple model, build step1 FitResult
        t0 = time.perf_counter()
        self.step1_solver.fit(xdata, pixel_signals, **fit_kwargs)
        step1_time = time.perf_counter() - t0

        # Temporarily swap solver to assemble step1 result
        _orig_solver = self.solver
        self.solver = self.step1_solver
        self.step1_result_ = self._assemble_fit_result(xdata, pixel_signals, step1_time)
        self.solver = _orig_solver

        # Step 2: fit complex model, fixing params from step 1 via fixed_param_maps
        fixed_maps = {name: self.step1_solver.params_[name] for name in self.fixed_from_step1}
        t0 = time.perf_counter()
        self.solver.fit(xdata, pixel_signals, pixel_fixed_params=fixed_maps, **fit_kwargs)
        fit_time = time.perf_counter() - t0

        for param, values in self.solver.params_.items():
            self.fitted_params_[param] = values

        # Assemble step-2 result, then replace fit_time with full wall-clock span
        step2_result = self._assemble_fit_result(xdata, pixel_signals, fit_time)
        total_elapsed = time.perf_counter() - t_total
        self.results_ = dataclasses.replace(step2_result, fit_time=total_elapsed)
        return self

    def predict(self, xdata, **predict_kwargs) -> np.ndarray:
        return super().predict(xdata, **predict_kwargs)
```

## Pyneapple solver interface (read-only — do not reimplement)

Fitters receive a solver instance. Key attributes/methods:

- `solver.model` — the model instance (ParametricModel or DistributionModel)
- `solver.fit(xdata, ydata, **kwargs) -> BaseSolver` — run optimization
- `solver.params_ -> dict[str, Any]` — fitted parameters (populated after `fit()`)
- `solver.diagnostics_ -> dict[str, Any]` — fit diagnostics (populated after `fit()`)
- `solver.pixel_results_ -> list[_PixelFitResult]` — per-pixel typed results consumed by `_assemble_fit_result()`

## Pyneapple model interface (read-only — do not reimplement)

Solvers and fitters access the model via `self.solver.model`. Key attributes/methods:

### ParametricModel
- `model.param_names -> list[str]` — free parameter names (excludes fixed)
- `model._all_param_names -> list[str]` — all parameter names (including fixed)
- `model.n_params -> int` — number of free parameters
- `model.forward(xdata, *params) -> np.ndarray` — evaluate the signal equation
- `model.jacobian(xdata, *params) -> np.ndarray | None` — analytical Jacobian (optional)
- `model.fixed_params -> dict[str, float]` — model-level fixed parameters
- `model.forward_with_fixed(xdata, fixed_dict, *free_params)` — forward with injected fixed values

### DistributionModel
- `model.bins -> np.ndarray` — shape (n_bins,), the discrete parameter grid
- `model.n_bins -> int` — number of bins
- `model.get_basis(xdata) -> np.ndarray` — shape (n_measurements, n_bins), the basis matrix
- `model.forward(xdata, *spectrum) -> np.ndarray` — reconstructs signal from spectrum coefficients

## Validation utilities

Pyneapple provides validation helpers in `pyneapple.utility.validation`:

- `validate_xdata(xdata)` — ensures xdata is a 1D numpy array
- `validate_data_shapes(xdata, image)` — ensures image last dimension matches xdata length
- `validate_segmentation(segmentation, image_shape)` — validates and normalises segmentation to 3D
- `validate_fixed_param_maps(maps, spatial_shape, param_names)` — validates per-pixel fixed parameter maps

## Coding conventions

- **Naming:** `PascalCase` classes, `snake_case` methods. Use descriptive suffixes like `PixelWiseFitter`, `SegmentationWiseFitter`, `IDEALFitter`
- **Fitted state:** trailing underscore: `fitted_params_`, `results_`, `step1_result_`
- **Logging:** use `loguru` via `from loguru import logger`; gate `logger.info()` behind `if self.verbose:`; never `logger.error()` then `raise` — just `raise`
- **Imports:** `from __future__ import annotations` in every module
- **Type hints:** `dict[str, np.ndarray]` for fitted params, `np.ndarray | None` for optional arrays
- **Error handling:** `ValueError` for bad inputs, `RuntimeError` for missing prerequisites, `ImportError` for optional deps
- **Tests:** pytest, class-based grouping, `pytest-mock` for mocking, docstrings on every test

## Entry_point registration

In `pyproject.toml`:

```toml
[project.entry-points."pyneapple.fitters"]
my_fitter = "my_package:MyFitterClass"
another_fitter = "my_package:AnotherFitterClass"
```

The key (e.g. `my_fitter`) is what users write in their TOML config:

```toml
[Fitting]
fitter = "my_fitter"
```
