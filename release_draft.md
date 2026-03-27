# v2.0.0-beta.1 (Pre-release)

> **⚠️ This is a pre-release. This major version introduces significant breaking changes to align with scikit-learn conventions.**

---

## What's New

### Architecture — scikit-learn Style Restructure

The entire codebase has been refactored to follow scikit-learn conventions with a clear separation of concerns:

- **Models**: Stateless, pure math forward model functions (`forward`, `jacobian`, `residual`)
- **Solvers**: Curve fitting backends (`CurveFitSolver`, `NNLSSolver`, `ConstrainedCurveFitSolver`)
- **Fitters**: High-level fitting orchestrators (`PixelWiseFitter`, `SegmentationWiseFitter`, `IDEALFitter`, `SegmentedFitter`)
- **Registry-based plugin discovery** via entry points for models, solvers, and fitters

### New I/O Module (`pyneapple.io`)

- `load_dwi_nifti`, `save_parameter_map`, `reconstruct_maps`, `save_spectrum_to_nifti`
- `load_bvalues`, `save_bvalues`
- `load_config` / `FittingConfig` for TOML configs
- Excel export: `save_params_to_excel`, `save_spectrum_to_excel`
- HDF5 export with gzip compression: `save_to_hdf5`, `load_from_hdf5`, `save_params_to_hdf5`

### New Utilities (`pyneapple.utility`)

- `spectrum` — Spectrum processing utilities
- `plotting` — Visualization helpers
- `validation` — Validation helpers

### New CLI Tools

```bash
pyneapple pixelwise         # Per-pixel fitting
pyneapple segmented         # Segmented fitting (two-step)
pyneapple ideal             # IDEAL iterative fitting
pyneapple segmentationwise  # ROI mean signal fitting
```

### Optional Dependencies

```bash
pip install pyneapple[excel,plotting,export]
```

---

## Breaking Changes

### 1. Removed Entire Module Hierarchy

| Old Path | Status |
|----------|--------|
| `pyneapple/parameters/` | **Removed entirely** |
| `pyneapple/results/` | **Removed entirely** |
| `pyneapple/fitting/` | **Removed entirely** |
| `pyneapple/utils/` | **Removed entirely** |

### 2. Removed Parameter Classes

```python
# REMOVED - No direct replacement
from pyneapple.parameters import Parameters
from pyneapple.parameters.ivim import IVIMParams
from pyneapple.parameters.ivim import IVIMSegmentedParams
from pyneapple.parameters.nnls import NNLSParams
from pyneapple.parameters.nnls import NNLSCVParams
from pyneapple.parameters.ideal import IDEALParams
from pyneapple.parameters.boundaries import BaseBoundaryDict
```

### 3. Removed Result Classes

```python
# REMOVED - No direct replacement
from pyneapple.results import Results
from pyneapple.results.ivim import IVIMResults
from pyneapple.results.ivim import IVIMSegmentedResults
from pyneapple.results.nnls import NNLSResults
from pyneapple.results.nnls import NNLSCVResults
from pyneapple.results.types import Results
from pyneapple.results.result_dict import ResultDict
```

### 4. Removed Fitting Interface

```python
# REMOVED - Replaced by new Fitters
from pyneapple.fitting.fitdata import FitData

# OLD API (removed)
fit_data = FitData(img, seg, "params.toml")
fit_data.fit_pixel_wise()
fit_data.fit_segmentation_wise()
fit_data.fit_ivim_segmented()
fit_data.fit_ideal()
```

### 5. Removed Utility Modules

```python
# REMOVED
from pyneapple.utils.logger import logger
from pyneapple.utils.plotting import plot_results
from pyneapple.utils.processing import *
```

### 6. Removed External Dependency

```python
# REMOVED - Was using radimgarray submodule
from radimgarray import RadImgArray, SegImgArray
```

### 7. TOML/JSON Parameter File Format — Complete Redesign

**Old format** (`params_biexp.toml`):
```toml
[General]
Class = "IVIMParams"          # ← Required class name
b_values = [0, 10, 50, ...]

[Model]
model = "biexp"
fit_s0 = false

[boundaries.D]
1 = [0.001, 0.0007, 0.05]    # ← Complex boundary format
2 = [0.02, 0.003, 0.3]
```

**New format** (`segmentationwise_biexp.toml`):
```toml
[Fitting]
fitter = "segmentationwise"

[Fitting.model]
type = "biexp"

[Fitting.solver]
type = "curvefit"
max_iter = 250

[Fitting.solver.p0]         # ← Simple initial guess
S0 = 1000.0
f1 = 0.2
D2 = 0.02

[Fitting.solver.bounds]     # ← Simple bounds
S0 = [1.0, 5000.0]
f1 = [0.01, 0.99]
```

### 8. CLI Interface Changes

```bash
# OLD
pyneapple-fitdata --params params_biexp.toml
pyneapple-fitdata --img data.nii --seg mask.nii

# NEW
pyneapple pixelwise --image data.nii --bval data.bval --config config.toml
pyneapple segmented --image data.nii --bval data.bval --config config.toml --seg mask.nii
pyneapple ideal --image data.nii --bval data.bval --config config.toml --seg mask.nii
```

### 9. Model API Changes

```python
# OLD - Multiple model classes for different fitting modes
from pyneapple.models import NNLSModel, NNLSCVModel
from pyneapple.models.ivim import BiExpFitModel, TriExpFitModel, MonoExpFitModel

# NEW - Unified model naming
from pyneapple.models import MonoExpModel, BiExpModel, TriExpModel, NNLSModel
```

### 10. Removed Example Scripts

- `examples/ivim_fitdata_script.py`
- `examples/nnls_fitdata_script.py`

### 11. Removed Example Parameter Files

All JSON and TOML parameter files in `examples/parameters/` have been replaced:
- `params_biexp.json`, `params_biexp.toml`
- `params_biexp_ideal.json`
- `params_biexp_segmented.json`
- `params_biexp_s0.toml`, `params_biexp_t1.toml`, `params_biexp_t1_steam.toml`
- `params_monoexp.json`
- `params_nnls.json`, `params_nnls.toml`, `params_nnls_cv.json`
- `params_triexp.json`, `params_triexp_ideal.json`, `params_triexp_reduced.toml`

### 12. Removed pygpufit Module

The `pygpufit` integration has been moved to a separate package.

---

## Import Changes Summary

| OLD | NEW |
|-----|-----|
| `Parameters` classes | `Models` |
| `IVIMParams`, `NNLSParams` | `BiExpModel`, `NNLSModel`, etc. |
| `Results` classes | No direct replacement |
| `FitData` | `Fitters` (`PixelWiseFitter`, etc.) |
| `logger` from `utils` | `configure_logging()` from `pyneapple` |
| `RadImgArray` | NIfTI via numpy arrays |
| `params.toml` (with `Class` field) | `config.toml` (`[Fitting]` section) |
| `fit_ivim()`, `fit_nnls()` | `fitter.fit(data, model)` |

---

## Migration Guide

### 1. Replace parameters with models

```python
# OLD
params = IVIMParams(bvalues=bvals, ...)
results = fit_ivim(data, params)

# NEW
from pyneapple.models import BiExpModel
from pyneapple.fitters import PixelWiseFitter
from pyneapple.solvers import NNLSSolver

model = BiExpModel(bvalues=bvals, ...)
fitter = PixelWiseFitter(solver=NNLSSolver())
results = fitter.fit(data, model)
```

### 2. Replace TOML configs

- Remove `Class` field
- Restructure under `[Fitting]` section
- Rename `[boundaries]` to `[Fitting.solver.bounds]`
- Add explicit `p0` section for initial guesses

### 3. Update imports

- `Parameters` → `Models`
- `Results` → Fitter return values (numpy arrays)
- `FitData` → `Fitters`

### 4. Update logging

```python
# OLD
from pyneapple.utils import logger
logger.info("message")

# NEW
from pyneapple import configure_logging
configure_logging(level="DEBUG")
```

---

## Installation

```bash
pip install pyneapple==2.0.0b1

# With optional dependencies
pip install pyneapple==2.0.0b1[excel,plotting,export]
```
