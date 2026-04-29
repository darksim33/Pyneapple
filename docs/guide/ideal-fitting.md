# IDEAL Fitting

> **TL;DR** — `IDEALFitter` performs iterative multi-resolution curve fitting. It starts with a coarse grid, fits the model, then interpolates parameters to a finer grid and refits. This coarse-to-fine approach improves convergence and reduces local minima. Covers the motivation and workflow, constructor arguments, configuration, TOML example, and Python API usage.

---

## Motivation

Multi-exponential diffusion models (bi-/tri-exponential) are nonlinear and sensitive to initial conditions. Poor initial guesses can cause convergence to local minima or divergence.

**IDEAL** (Image Downsampling Expedited Adaptive Least‐squares) addresses this by:

1. Starting at a coarse grid (e.g., 2×2 voxels)
2. Fitting the model to get initial parameter estimates
3. Interpolating parameter maps to the next finer resolution
4. Using those interpolated values as initial guesses for the next fit
5. Repeating until reaching the full image resolution

This coarse-to-fine approach:
- Provides physically reasonable initial guesses at each level
- Reduces sensitivity to local minima
- Improves robustness of perfusion (IVIM) parameter estimates

---

## Workflow overview

```
image (full resolution)
  │
  ├─▶ Step 1: Interpolate to coarse grid (dim_steps[0])
  │          Fit model → get parameter map
  │
  ├─▶ Step 2: Interpolate param map to finer grid (dim_steps[1])
  │          Use interpolated params as p0 + scaled bounds
  │          Fit model → refined parameter map
  │
  ├─▶ Step 3: ... (repeat for each row in dim_steps)
  │
  └─▶ Final: Full-resolution parameter map
```

At each step after the first, bounds are scaled by `step_tol` around the interpolated parameters.

---

## Constructor

```python
from pyneapple.fitters import IDEALFitter
from pyneapple.solvers import CurveFitSolver
from pyneapple.models import BiExpModel

solver = CurveFitSolver(
    model=BiExpModel(),
    max_iter=250,
    tol=1e-8,
    p0={"S0": 1000, "f1": 0.2, "D1": 0.001, "D2": 0.02},
    bounds={"S0": [1, 5000], "f1": [0.01, 0.99], "D1": [1e-5, 0.003], "D2": [0.003, 0.3]},
)

fitter = IDEALFitter(
    solver=solver,
    dim_steps=np.array([[16, 16], [32, 32], [64, 64], [128, 128]]),
    step_tol={"S0": 0.5, "f1": 0.2, "D1": 0.2, "D2": 0.2},
    ideal_dims=2,
    segmentation_threshold=0.2,
    interpolation_method="cubic",
)
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `solver` | `CurveFitSolver` | *required* | Solver with configured model, p0, bounds |
| `dim_steps` | `np.ndarray` | *required* | 2D array shape `(n_steps, ideal_dims)` — each row is a step, e.g. `[[16,16], [32,32], [64,64]]` |
| `step_tol` | `dict[str, float]` | *required* | Tolerance per parameter — keys must match model `param_names`, values scale bounds at each level |
| `ideal_dims` | `int` | 2 | Number of spatial dimensions in the grid (2 or 3) |
| `segmentation_threshold` | `float` | 0.2 | Threshold (0–1) for including pixels in fitting |
| `interpolation_method` | `str` | "cubic" | Interpolation method — `"linear"` or `"cubic"` |
| `**fitter_kwargs` | `dict` | `{}` | Additional arguments passed to `BaseFitter` |

### Validation rules

- `dim_steps` must be 2D with `ideal_dims` columns
- Each row must increase monotonically (coarse → fine)
- The last row must match the image spatial dimensions
- `step_tol` keys must exactly match `solver.model.param_names`

---

## Accessing results

After fitting:

```python
fitter.fit(bvalues, image_4d)

# Get fitted parameters as a dict of spatial maps
params = fitter.get_fitted_params()
# Returns: {"S0": np.ndarray, "f1": np.ndarray, "D1": np.ndarray, "D2": np.ndarray}

# Predict signal at new b-values
predictions = fitter.predict(bvalues_new)
# Returns: np.ndarray of shape (X, Y, Z, len(bvalues_new))
```

---

## TOML configuration

```toml
[Fitting]
fitter = "ideal"

[Fitting.model]
type   = "biexp"
fit_s0 = false

[Fitting.solver]
type     = "curvefit"
max_iter = 250
tol      = 1e-8

[Fitting.solver.p0]
S0  = 1000.0
f1  = 0.2
D2  = 0.02
D1  = 0.001

[Fitting.solver.bounds]
S0  = [1.0,   5000.0]
f1  = [0.01,  0.99]
D2  = [0.003, 0.3]
D1  = [1e-5,  0.003]

[Fitting.ideal]
dim_steps              = [[16, 16], [32, 32], [64, 64], [128, 128]]
ideal_dims             = 2
segmentation_threshold = 0.2
interpolation_method   = "cubic"

[Fitting.ideal.step_tol]
S0  = 0.5
f1  = 0.2
D2  = 0.2
D1  = 0.2
```

### CLI usage

```bash
pyneapple-ideal \
    --image  dwi.nii.gz \
    --bval   dwi.bval \
    --config ideal_biexp.toml \
    --seg    mask.nii.gz
```

---

## Python API example

```python
import numpy as np
from pyneapple.fitters import IDEALFitter
from pyneapple.solvers import CurveFitSolver
from pyneapple.models import BiExpModel
from pyneapple.io import load_dwi_nifti, load_bvalues

# Load data
image, _ = load_dwi_nifti("dwi.nii.gz")
bvalues = load_bvalues("dwi.bval")

# Configure solver
solver = CurveFitSolver(
    model=BiExpModel(),
    max_iter=250,
    tol=1e-8,
    p0={"S0": 1000, "f1": 0.2, "D1": 0.001, "D2": 0.02},
    bounds={"S0": [1, 5000], "f1": [0.01, 0.99], "D1": [1e-5, 0.003], "D2": [0.003, 0.3]},
)

# Configure IDEAL fitter
fitter = IDEALFitter(
    solver=solver,
    dim_steps=np.array([[16, 32, 64, 128], [16, 32, 64, 128]]),
    step_tol={"S0": 0.5, "f1": 0.2, "D1": 0.2, "D2": 0.2},
    ideal_dims=2,
)

# Run fit
fitter.fit(bvalues, image)

# Get results
params = fitter.get_fitted_params()
print("D2 map:", params["D2"])
```

---

## Tips and caveats

1. **Choose `dim_steps` carefully** — The first step should be coarse (e.g., 2×2) to provide global initial estimates. The last step must match your image dimensions.

2. **Adjust `step_tol`** — Tighter tolerances (smaller values) in later steps allow refinement. Larger tolerances in early steps provide wider bounds for robustness.

