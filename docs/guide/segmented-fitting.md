# Segmented Fitting

> **TL;DR** — `SegmentedFitter` runs a two-step fitting pipeline: fit a simple model on a b-value subset to estimate baseline parameters, then fit a complex model on the full b-value range with those parameters fixed. Covers the motivation and workflow, constructor arguments and validation rules, b-value subsetting, parameter fixing and mapping, accessing results, and a complete worked example.

---

## Motivation

Multi-exponential diffusion models (bi-/tri-exponential) are sensitive to initial conditions and often converge to local minima. A common strategy is to split the fit into two stages:

1. **Step 1** — Fit a monoexponential model on high b-values (e.g. b >= 200 s/mm²) to obtain a stable estimate of the slow diffusion coefficient (ADC).
2. **Step 2** — Fit the full multi-exponential model on all b-values, fixing the slow diffusion coefficient to the Step 1 estimate.

This reduces the degrees of freedom in Step 2, stabilizes convergence, and produces more reproducible perfusion fraction estimates.

`SegmentedFitter` encapsulates this pattern as a single API call.

---

## Workflow overview

```
xdata (all b-values)
  │
  ├──▶ _subset_bvalues() ──▶ Step 1: PixelWiseFitter + MonoExpModel
  │                                    │
  │                             step1_params_ (e.g. S0, D)
  │                                    │
  │    ┌───────────────────────────────┘
  │    │  _reconstruct_volume()
  │    │  param_mapping: D → D2
  │    ▼
  └──▶ Step 2: PixelWiseFitter + BiExpModel (D2 fixed per-pixel)
                        │
                 fitted_params_ (f1, D1, D2)
```

Both steps use `PixelWiseFitter` internally. Step 1 results are passed to Step 2 as `fixed_param_maps`, so each pixel receives its own fixed value.

---

## Constructor

```python
from pyneapple.fitters import SegmentedFitter

fitter = SegmentedFitter(
    step1_solver=solver1,        # CurveFitSolver with MonoExpModel
    step2_solver=solver2,        # CurveFitSolver with BiExpModel
    step1_bvalue_range=(200, None),  # b >= 200 for Step 1
    fixed_from_step1=["D"],      # fix D from Step 1 in Step 2
    param_mapping={"D": "D2"},   # MonoExp D → BiExp D2
)
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `step1_solver` | `BaseSolver` | *required* | Solver for Step 1 (simple model) |
| `step2_solver` | `BaseSolver` | *required* | Solver for Step 2 (complex model) |
| `step1_bvalue_range` | `tuple[float\|None, float\|None] \| None` | `None` | `(lo, hi)` range filter for Step 1 b-values. `None` for open end. |
| `fixed_from_step1` | `list[str] \| None` | `[]` | Step 1 parameter names to fix in Step 2 |
| `param_mapping` | `dict[str, str] \| None` | `{}` | Maps Step 1 names to Step 2 names (identity if absent) |

### Validation rules

The constructor validates immediately:

- Every name in `fixed_from_step1` must be a parameter of the Step 1 model.
- Every mapped target name must be a parameter of the Step 2 model.

At fit time:

- At least 3 b-values must fall within `step1_bvalue_range`.
- Standard `xdata` and `image` shape checks apply.

---

## B-value subsetting

`step1_bvalue_range` selects which b-values are used in Step 1. Step 2 always uses all b-values.

| `step1_bvalue_range` | Effect |
|---|---|
| `None` | All b-values (no filtering) |
| `(200, None)` | b >= 200 |
| `(None, 500)` | b <= 500 |
| `(200, 800)` | 200 <= b <= 800 |

The corresponding image slices are extracted automatically. A `ValueError` is raised if the range selects fewer than 3 b-values.

---

## Parameter fixing and mapping

`fixed_from_step1` controls which Step 1 fitted parameters become per-pixel constants in Step 2.

`param_mapping` handles the case where the two models use different names for the same physical quantity. For example, `MonoExpModel` calls the diffusion coefficient `D`, while `BiExpModel` (reduced) calls the slow component `D2`:

```python
fixed_from_step1=["D"],
param_mapping={"D": "D2"},
```

Parameters not listed in the mapping are assumed to share the same name. Omitting `fixed_from_step1` runs both steps independently (no parameter passing).

---

## Fitting

```python
fitter.fit(xdata=bvalues, image=image_4d, segmentation=mask_3d)
```

| Argument | Type | Description |
|---|---|---|
| `xdata` | `np.ndarray` shape `(N,)` | Full b-value array |
| `image` | `np.ndarray` shape `(X, Y, Z, N)` | 4-D DWI volume |
| `segmentation` | `np.ndarray` shape `(X, Y, Z)` or `None` | Binary mask (non-zero = include) |

Returns `self` for chaining.

---

## Accessing results

### Fitted parameters

```python
params = fitter.get_fitted_params()
# {"f1": ndarray (n_pixels,), "D1": ndarray (n_pixels,), "D2": ndarray (n_pixels,)}
```

The dict contains all Step 2 free parameters plus any fixed parameters merged back from Step 1.

### Step 1 parameters

```python
step1 = fitter.step1_params_
# {"S0": ndarray (n_pixels,), "D": ndarray (n_pixels,)}
```

### Prediction

```python
reconstructed = fitter.predict(xdata=bvalues)
# np.ndarray shape (X, Y, Z, N)
```

Uses the Step 2 model with the combined (free + fixed) parameters.

---

## Complete example

```python
import numpy as np
from pyneapple.models import MonoExpModel, BiExpModel
from pyneapple.solvers import CurveFitSolver
from pyneapple.fitters import SegmentedFitter
from pyneapple.io import load_dwi_nifti, load_bvalues, save_parameter_map

# Load data
image, nifti_ref = load_dwi_nifti("dwi.nii.gz")
bvalues = load_bvalues("dwi.bval")
mask = np.ones(image.shape[:3], dtype=bool)  # or load a segmentation

# Step 1: MonoExp on high b-values for ADC estimate
solver1 = CurveFitSolver(
    model=MonoExpModel(),
    p0={"S0": 1.0, "D": 0.001},
    bounds={"S0": (0.01, 5.0), "D": (1e-5, 0.1)},
    max_iter=250,
)

# Step 2: BiExp (reduced) on all b-values, D2 fixed from Step 1
solver2 = CurveFitSolver(
    model=BiExpModel(fit_reduced=True),
    p0={"f1": 0.2, "D1": 0.01, "D2": 0.001},
    bounds={"f1": (0.0, 1.0), "D1": (0.001, 0.1), "D2": (1e-5, 0.01)},
    max_iter=500,
)

# Build and run
fitter = SegmentedFitter(
    step1_solver=solver1,
    step2_solver=solver2,
    step1_bvalue_range=(200, None),
    fixed_from_step1=["D"],
    param_mapping={"D": "D2"},
)
fitter.fit(xdata=bvalues, image=image, segmentation=mask)

# Save results
for name, values in fitter.get_fitted_params().items():
    # Reconstruct spatial volume for saving
    vol = fitter._reconstruct_volume(values, fitter.pixel_indices, image.shape[:3])
    save_parameter_map(params=vol, path=f"results/{name}.nii.gz", reference_nifti=nifti_ref)
```

---

## Without parameter fixing

Run two independent steps (e.g. for comparison) by omitting `fixed_from_step1`:

```python
fitter = SegmentedFitter(
    step1_solver=solver1,
    step2_solver=solver2,
    step1_bvalue_range=(200, None),
)
fitter.fit(xdata=bvalues, image=image)

# Both step sets are available
print(fitter.step1_params_.keys())   # {'S0', 'D'}
print(fitter.get_fitted_params().keys())  # {'f1', 'D1', 'D2'}
```
