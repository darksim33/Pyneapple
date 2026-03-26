# Python API

> **TL;DR** — Use Pyneapple programmatically without the CLI by building a model, solver, and fitter directly in Python. Covers a minimal end-to-end fit, loading NIfTI and b-value inputs, constructing each layer individually, fixing model parameters (scalar or per-pixel maps), and accessing fitted parameters and diagnostics.

---

## End-to-end example

```python
from pyneapple import MonoExpModel, CurveFitSolver, PixelWiseFitter
from pyneapple.io import load_dwi_nifti, load_bvalues
import numpy as np

# Load inputs
image, nifti_ref = load_dwi_nifti("dwi.nii.gz")  # shape (X, Y, Z, N)
bvalues = load_bvalues("dwi.bval")               # shape (N,)

# Build the stack
model  = MonoExpModel()
solver = CurveFitSolver(
    model=model,
    p0={"S0": 1000.0, "D": 0.001},
    bounds={"S0": (1.0, 5000.0), "D": (1e-5, 0.1)},
    max_iter=250,
)
fitter = PixelWiseFitter(solver=solver)

# Fit
fitter.fit(xdata=bvalues, image=image)

# Results — dict of {param_name: ndarray shape (X, Y, Z)}
params = fitter.get_fitted_params()
print(params.keys())  # dict_keys(['S0', 'D'])
```

---

## Loading inputs

### NIfTI image

```python
from pyneapple.io import load_dwi_nifti

image, nifti_ref = load_dwi_nifti("subject01.nii.gz")
# image: np.ndarray shape (X, Y, Z, N_b)
# nifti_ref: nibabel.Nifti1Image — use as reference when saving outputs
```

### B-values

```python
from pyneapple.io import load_bvalues

bvalues = load_bvalues("subject01.bval")
# np.ndarray shape (N_b,), values in s/mm²
```

### Config file (optional)

If you already have a TOML config, `FittingConfig.build_fitter()` constructs the full stack for you:

```python
from pyneapple.io import load_config

config = load_config("config.toml")
fitter = config.build_fitter()      # returns PixelWiseFitter, ready to use
fitter.fit(xdata=bvalues, image=image)
```

---

## Models

All models share a common interface:

```python
model.forward(xdata, *params)   # → np.ndarray signal, shape (N,)
model.jacobian(xdata, *params)  # → np.ndarray | None
model.residual(xdata, signal, params)  # → np.ndarray
model.param_names               # list[str] — ordered parameter names
```

Construct with keyword arguments matching the TOML model keys:

```python
from pyneapple import BiExpModel, NNLSModel

# Reduced (default) — params: f1, D1, D2
biexp = BiExpModel()

# Full fractions — params: f1, D1, f2, D2
biexp_full = BiExpModel(fit_reduced=False)

# With T1 correction — params: S0, D, T1
from pyneapple import MonoExpModel
mono_t1 = MonoExpModel(fit_t1=True, repetition_time=3000.0)

# NNLS distribution
nnls_model = NNLSModel(d_range=(0.0008, 0.5), n_bins=250)
basis = nnls_model.get_basis(bvalues)  # shape (N_b, 250)
bins  = nnls_model.bins                # shape (250,)
```

---

## Solvers

### `CurveFitSolver`

Wraps `scipy.optimize.curve_fit`. Use for parametric models (`MonoExpModel`, `BiExpModel`, `TriExpModel`).

```python
from pyneapple import CurveFitSolver, MonoExpModel

solver = CurveFitSolver(
    model=MonoExpModel(),
    p0={"S0": 1000.0, "D": 0.001},         # initial guesses
    bounds={"S0": (1.0, 5000.0), "D": (1e-5, 0.1)},
    max_iter=250,
    tol=1e-8,
    multi_threading=True,
    n_pools=-1,  # all cores
)

solver.fit(xdata=bvalues, ydata=signal)   # signal shape (N_pixels, N_b)

solver.get_params()       # {"S0": np.ndarray, "D": np.ndarray}
solver.get_diagnostics()  # {"pcov": np.ndarray}  — covariance matrix
```

### `NNLSSolver`

Wraps `scipy.optimize.nnls` with optional Tikhonov regularisation. Use with `NNLSModel`.

```python
from pyneapple.solvers import NNLSSolver
from pyneapple import NNLSModel

solver = NNLSSolver(
    model=NNLSModel(d_range=(0.0008, 0.5), n_bins=250),
    reg_order=2,   # 0 = none, 1 = first diff, 2 = second diff, 3 = curvature
    mu=0.02,
    multi_threading=True,
    n_pools=4,
)

solver.fit(xdata=bvalues, signal=signal)  # signal shape (N_pixels, N_b)

solver.get_params()       # {"coefficients": np.ndarray shape (N_pixels, N_bins)}
solver.get_diagnostics()  # {"residual": np.ndarray}
```

---

## Fitters

### `PixelWiseFitter`

Iterates over all non-zero voxels and dispatches each to the solver.

```python
from pyneapple import PixelWiseFitter

fitter = PixelWiseFitter(solver=solver)

# Fit all voxels
fitter.fit(xdata=bvalues, image=image)

# Fit only voxels inside a mask
fitter.fit(xdata=bvalues, image=image, segmentation=mask)
# mask: np.ndarray shape (X, Y, Z), non-zero = include

# Retrieve results
params = fitter.get_fitted_params()
# {"S0": np.ndarray (X, Y, Z), "D": np.ndarray (X, Y, Z), ...}

# Reconstruct signal from fitted parameters
reconstructed = fitter.predict(xdata=bvalues)
# np.ndarray shape (X, Y, Z, N_b)
```

---

## Saving outputs

```python
from pyneapple.io import save_parameter_map

for param_name, param_map in fitter.get_fitted_params().items():
    save_parameter_map(
        params=param_map,
        path=f"results/dwi_{param_name}.nii.gz",
        reference_nifti=nifti_ref,
    )
```

---

## Fixed parameters

Fix one or more model parameters so they are held constant during fitting. Pyneapple supports two levels: scalar (model-wide) and per-pixel maps.

### Scalar fixed parameters

Pass `fixed_params` when constructing the model. Every voxel uses the same constant value.

```python
from pyneapple import MonoExpModel, CurveFitSolver, PixelWiseFitter

model = MonoExpModel(fit_t1=True, repetition_time=3000.0, fixed_params={"T1": 1000.0})
print(model.param_names)  # ['S0', 'D'] — T1 is excluded from fitting

solver = CurveFitSolver(
    model=model,
    p0={"S0": 1000.0, "D": 0.001},
    bounds={"S0": (1.0, 5000.0), "D": (1e-5, 0.1)},
)
fitter = PixelWiseFitter(solver=solver)
fitter.fit(xdata=bvalues, image=image)
```

### Per-pixel fixed parameter maps

Pass `fixed_param_maps` to `fitter.fit()` to supply a spatially varying value for each voxel. Each map must match the spatial shape of the image `(X, Y, Z)`.

```python
import numpy as np
from pyneapple import BiExpModel, CurveFitSolver, PixelWiseFitter

model = BiExpModel()
solver = CurveFitSolver(
    model=model,
    p0={"f1": 0.3, "D1": 0.01, "D2": 0.001},
    bounds={"f1": (0.0, 1.0), "D1": (1e-4, 0.1), "D2": (1e-5, 0.01)},
)
fitter = PixelWiseFitter(solver=solver)

# Load or create a per-pixel map — shape must be (X, Y, Z)
d2_map = np.full(image.shape[:3], 0.003)

fitter.fit(xdata=bvalues, image=image, fixed_param_maps={"D2": d2_map})
# Only f1 and D1 are fitted; D2 is read from d2_map at each voxel
```

Per-pixel maps override any scalar `fixed_params` set on the model for the same parameter name.
