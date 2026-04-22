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
   - Delegate optimization to `self.solver.fit(...)`
   - Store fitted parameters in `self.fitted_params_` (a `dict[str, np.ndarray]`)
   - Return `self`

### BaseFitter interface

```python
class BaseFitter(ABC):
    def __init__(self, solver: BaseSolver, verbose: bool = False, **fitter_kwargs):
        self.solver = solver
        self.verbose = verbose
        self.fitter_kwargs = fitter_kwargs
        self.results_: Any = None
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
    def _check_fitted(self) -> None: ...        # raises RuntimeError if not fitted
    def _get_param_names(self) -> list[str]: ... # delegates to self.solver.model.param_names
```

### Pixel-wise fitter pattern

For fitting each pixel independently (parametric or distribution models):

```python
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

        pixel_to_fit = self._extract_pixel_data(image, segmentation)

        # Delegate to solver
        self.solver.fit(xdata, pixel_to_fit, **fit_kwargs)

        # Store results
        for param, values in self.solver.params_.items():
            self.fitted_params_[param] = values

        return self

    def predict(self, xdata, **predict_kwargs) -> np.ndarray:
        return super().predict(xdata, **predict_kwargs)
```

### Segmentation-wise fitter pattern

For fitting mean signal per segmented region:

```python
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
        # Fit on aggregated signals
        # Store results in self.fitted_params_
        return self

    def predict(self, xdata, **predict_kwargs) -> np.ndarray:
        # Override to broadcast segment-level predictions back to pixel level
        ...
```

### Multi-step pipeline fitter pattern

For orchestrating multiple solvers in sequence (e.g., simple model then complex model):

```python
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
        self.step1_params_: dict[str, np.ndarray] = {}

    def fit(self, xdata, image, segmentation=None, **fit_kwargs) -> "MyPipelineFitter":
        # Step 1: fit simple model (optionally on b-value subset)
        # Step 2: fit complex model, fixing params from Step 1 via fixed_param_maps
        # Store combined results in self.fitted_params_
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
- **Fitted state:** trailing underscore: `fitted_params_`, `results_`, `step1_params_`
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
