# Copilot Instructions — Pyneapple Plugin Template

This is a **Pyneapple plugin** that provides solvers for diffusion MRI analysis.
It registers into Pyneapple's entry_point-based plugin system.

## What this package does

- Provides one or more solvers (CPU, GPU, or hybrid implementations)
- Each solver inherits from `pyneapple.solvers.base.BaseSolver`
- Registers via `[project.entry-points."pyneapple.solvers"]` in `pyproject.toml`
- Heavy dependencies (CUDA, proprietary libs, etc.) are only imported when a solver is instantiated, never at discovery time

## Pyneapple solver contract

Every solver **must**:

1. Inherit from `pyneapple.solvers.base.BaseSolver`
2. Call `super().__init__(model=model, max_iter=max_iter, tol=tol, verbose=verbose, **solver_kwargs)`
3. Implement `fit(self, xdata, ydata, **kwargs) -> self`
4. Inside `fit()`:
   - Call `self._reset_state()` first
   - Store fitted parameters in `self.params_` (a `dict[str, Any]`)
   - Store diagnostics in `self.diagnostics_` (a `dict[str, Any]`)
   - Return `self`
5. Implement `_fit_single_pixel()` to return a `_PixelFitResult` (see below)
6. After fitting all pixels, store the list of per-pixel results in `self.pixel_results_`

### BaseSolver interface

```python
from pyneapple.solvers.base import BaseSolver, _PixelFitResult

class BaseSolver(ABC):
    def __init__(self, model, max_iter=250, tol=1e-8, verbose=False, **solver_kwargs):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.diagnostics_: dict[str, Any] = {}
        self.params_: dict[str, Any] = {}
        self.pixel_results_: list[_PixelFitResult] = []  # populated by fit()

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseSolver": ...

    def get_diagnostics(self) -> dict[str, Any]: ...  # raises RuntimeError if empty
    def get_params(self) -> dict[str, Any]: ...        # raises RuntimeError if empty
    def _reset_state(self): ...                        # clears params_, diagnostics_, pixel_results_
```

### _PixelFitResult dataclass

Every `_fit_single_pixel()` implementation must return a `_PixelFitResult`.
This is a private dataclass defined in `pyneapple.solvers.base`:

```python
@dataclass
class _PixelFitResult:
    params: np.ndarray          # 1-D array of fitted values for one pixel (required)
    covariance: np.ndarray | None = None   # (n_params, n_params) or None
    success: bool = True                   # False if the optimiser failed for this pixel
    message: str | None = None             # optimiser status message, if available
    n_iterations: int | None = None        # iteration count, if the backend exposes it
    residual: float | None = None          # scalar residual norm, if available
```

Only populate the fields your backend actually provides — leave all others as `None`.
The fitter's `_assemble_fit_result()` handles `None` gracefully in every field.

### Parametric solver pattern

For fitting parametric models (e.g., mono-exponential, bi-exponential, tri-exponential):

```python
from pyneapple.solvers.base import BaseSolver, _PixelFitResult

class MyCurveFitSolver(BaseSolver):
    def __init__(self, model, max_iter, tol,
                 p0: dict[str, float] | None = None,
                 bounds: dict[str, tuple[float, float]] | None = None,
                 verbose=False, **solver_kwargs):
        super().__init__(model=model, max_iter=max_iter, tol=tol, verbose=verbose)
        # Validate p0 keys match model.param_names
        # Validate bounds keys match model.param_names
        # Store implementation-specific options

    def fit(self, xdata, ydata, p0=None, bounds=None,
            pixel_fixed_params=None, **fit_kwargs) -> "MyCurveFitSolver":
        self._reset_state()
        # xdata: 1D np.ndarray (e.g. b-values)
        # ydata: 2D np.ndarray shape (n_pixels, n_xdata)

        pixel_results: list[_PixelFitResult] = []
        for i in range(n_pixels):
            pr = self._fit_single_pixel(xdata, ydata[i], p0[:, i], bounds_i)
            pixel_results.append(pr)

        # Store per-pixel results for FitResult assembly by the fitter
        self.pixel_results_ = pixel_results

        # Unpack into backwards-compatible storage
        self.params_ = {name: np.array([pr.params[j] for pr in pixel_results])
                        for j, name in enumerate(self.model.param_names)}
        self.diagnostics_ = {
            "pcov": np.array([pr.covariance if pr.covariance is not None
                              else np.full((n_params, n_params), np.nan)
                              for pr in pixel_results]),
            "n_pixels": n_pixels,
        }
        return self

    def _fit_single_pixel(
        self, xdata, ydata, p0, bounds, pixel_idx=None, pixel_fixed=None
    ) -> _PixelFitResult:
        try:
            popt, pcov = curve_fit(self.model.forward, xdata, ydata,
                                   p0=p0, bounds=bounds, ...)
            return _PixelFitResult(params=popt, covariance=pcov, success=True)
        except RuntimeError as exc:
            return _PixelFitResult(
                params=p0,
                covariance=np.full((len(p0), len(p0)), np.nan),
                success=False,
                message=str(exc),
            )
```

### Distribution solver pattern

For fitting distribution models (e.g., NNLS):

```python
from pyneapple.solvers.base import BaseSolver, _PixelFitResult

class MyNNLSSolver(BaseSolver):
    def __init__(self, model: DistributionModel, reg_order=0, mu=0.02,
                 max_iter=250, tol=1e-8, verbose=False,
                 **solver_kwargs):
        super().__init__(model, max_iter, tol, verbose, **solver_kwargs)
        # model provides: model.bins, model.n_bins, model.get_basis(xdata)

    def fit(self, xdata, signal, pixel_fixed_params=None) -> "MyNNLSSolver":
        self._reset_state()

        pixel_results: list[_PixelFitResult] = []
        for i in range(n_pixels):
            pr = self._fit_single_pixel(basis, signal[i], pixel_idx=i)
            pixel_results.append(pr)

        # Store per-pixel results for FitResult assembly by the fitter
        self.pixel_results_ = pixel_results

        # Unpack into backwards-compatible storage
        self.params_["coefficients"] = np.array([pr.params for pr in pixel_results])
        self.diagnostics_["residuals"] = np.array([
            pr.residual if pr.residual is not None else np.nan
            for pr in pixel_results
        ])
        return self

    def _fit_single_pixel(self, basis, ydata, pixel_idx=None) -> _PixelFitResult:
        try:
            coeffs, residual = nnls(basis, ydata)
            return _PixelFitResult(params=coeffs, residual=float(residual), success=True)
        except Exception as exc:
            return _PixelFitResult(
                params=np.zeros(self.model.n_bins),
                residual=float(np.linalg.norm(ydata)),
                success=False,
                message=str(exc),
            )
```

## Pyneapple model interface (read-only — do not reimplement)

Solvers receive a model instance. Key attributes/methods:

### ParametricModel
- `model.param_names -> list[str]` — free parameter names (excludes fixed)
- `model.n_params -> int` — number of free parameters
- `model.forward(xdata, *params) -> np.ndarray` — evaluate the signal equation
- `model.jacobian(xdata, *params) -> np.ndarray | None` — analytical Jacobian (optional)
- `model.fixed_params -> dict[str, float]` — model-level fixed parameters
- `model.forward_with_fixed(xdata, fixed_dict, *free_params)` — forward with injected fixed values
- `model.jacobian_with_fixed(xdata, fixed_dict, *free_params)` — Jacobian with fixed values sliced

### DistributionModel
- `model.bins -> np.ndarray` — shape (n_bins,), the discrete parameter grid
- `model.n_bins -> int` — number of bins
- `model.get_basis(xdata) -> np.ndarray` — shape (n_measurements, n_bins), the basis matrix
- `model.forward(xdata, *spectrum) -> np.ndarray` — reconstructs signal from spectrum coefficients

## Coding conventions

- **Naming:** `PascalCase` classes, `snake_case` methods. Use descriptive suffixes like `CurveFitSolver`, `NNLSSolver`, `TGV2Solver`
- **Fitted state:** trailing underscore: `params_`, `diagnostics_`, `pixel_results_`
- **Logging:** use `loguru` via `from loguru import logger`; gate `logger.info()` behind `if self.verbose:`; never `logger.error()` then `raise` — just `raise`
- **Imports:** `from __future__ import annotations` in every module
- **Type hints:** `dict[str, float | np.ndarray]` for params, `dict[str, tuple[float, float]] | None` for bounds
- **Error handling:** `ValueError` for bad inputs, `RuntimeError` for missing prerequisites, `ImportError` for optional deps
- **Tests:** pytest, class-based grouping, `pytest-mock` for mocking, docstrings on every test

## Entry_point registration

In `pyproject.toml`:

```toml
[project.entry-points."pyneapple.solvers"]
my_solver = "my_package:MySolverClass"
another_solver = "my_package:AnotherSolverClass"
```

The key (e.g. `my_solver`) is what users write in their TOML config:

```toml
[Fitting.solver]
type = "my_solver"
```
