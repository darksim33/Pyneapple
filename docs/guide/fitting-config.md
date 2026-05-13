# FittingConfig

> **TL;DR** — `load_config` reads a TOML file and returns a `FittingConfig` that assembles the full model → solver → fitter stack with one call. Covers the three-line workflow, the `FittingConfig` attribute reference, error cases, and complete TOML + Python examples for each fitter type.

---

## Overview

`FittingConfig` is the third way to start a fit in Pyneapple — alongside the [CLI](cli.md) and [explicit Python construction](python-api.md). It is the right choice when:

- A TOML config file already exists (e.g. shared with collaborators).
- You want to run a fit from a script without repeating all constructor arguments.
- You want to change fitting parameters without editing Python code.

The full workflow is three lines:

```python
from pyneapple.io import load_config, load_dwi_nifti, load_bvalues

image, nifti_ref = load_dwi_nifti("dwi.nii.gz")
bvalues          = load_bvalues("dwi.bval")

config = load_config("config.toml")   # parse TOML → FittingConfig
fitter = config.build_fitter()        # build model + solver + fitter

fitter.fit(xdata=bvalues, image=image)
params = fitter.get_fitted_params()   # {"S0": ndarray, "D": ndarray, ...}
```

---

## `load_config`

```python
from pyneapple.io import load_config

config = load_config("config.toml")   # str or pathlib.Path
```

Reads and validates the file. Raises on the first error rather than collecting them, so the message always points to the offending key.

| Raises | Condition |
|---|---|
| `FileNotFoundError` | Path does not exist |
| `KeyError` | `[Fitting]` section is missing |
| `ValueError` | Unknown `fitter`, `model`, or `solver` type; malformed bounds |

---

## `FittingConfig` attributes

`FittingConfig` is a plain dataclass. Inspect or override its fields before calling `build_fitter()`.

| Attribute | Type | Description |
|---|---|---|
| `fitter_type` | `str` | Registered fitter name — `"pixelwise"`, `"ideal"`, `"segmented"`, `"segmentationwise"` |
| `model_type` | `str` | Registered model name — `"monoexp"`, `"biexp"`, `"triexp"`, `"nnls"` |
| `solver_type` | `str` | Registered solver name — `"curvefit"`, `"nnls"`, `"constrained_curvefit"` |
| `model_kwargs` | `dict` | Extra kwargs forwarded to the model constructor |
| `solver_kwargs` | `dict` | Extra kwargs forwarded to the solver constructor |
| `p0` | `dict[str, float]` | Initial parameter guesses |
| `bounds` | `dict[str, tuple[float, float]]` | Per-parameter `(lower, upper)` bounds |
| `fixed_params` | `dict[str, float]` | Scalar parameters held constant during fitting |
| `ideal_kwargs` | `dict` | IDEAL-specific keys from `[Fitting.ideal]` (empty for other fitters) |
| `segmented_kwargs` | `dict` | Segmented-specific keys from `[Fitting.segmented]` (empty for other fitters) |

Override before building — for example, to tighten bounds programmatically:

```python
config = load_config("config.toml")
config.bounds["D"] = (1e-4, 0.05)   # narrower than the TOML value
fitter = config.build_fitter()
```

---

## `build_fitter`

```python
fitter = config.build_fitter()
```

Instantiates the model, solver, and fitter in order. Returns a fully configured `BaseFitter` — call `.fit()` on it directly.

| Raises | Condition |
|---|---|
| `KeyError` | A registered type is missing from its registry |
| `ValueError` | IDEAL fitter requested but `[Fitting.ideal]` section is absent |

---

## Examples

### Pixelwise (mono-exponential)

**TOML**

```toml
[Fitting]
fitter = "pixelwise"

[Fitting.model]
type = "monoexp"

[Fitting.solver]
type     = "curvefit"
max_iter = 250
tol      = 1e-8

[Fitting.solver.p0]
S0 = 1000.0
D  = 0.001

[Fitting.solver.bounds]
S0 = [1.0, 5000.0]
D  = [1e-5, 0.1]
```

**Python**

```python
from pyneapple.io import load_config, load_dwi_nifti, load_bvalues, save_parameter_map

image, nifti_ref = load_dwi_nifti("dwi.nii.gz")
bvalues          = load_bvalues("dwi.bval")

fitter = load_config("monoexp.toml").build_fitter()
fitter.fit(xdata=bvalues, image=image)

for name, param_map in fitter.get_fitted_params().items():
    save_parameter_map(
        params=param_map,
        path=f"results/dwi_{name}.nii.gz",
        reference_nifti=nifti_ref,
    )
```

---

### IDEAL (bi-exponential)

**TOML**

```toml
[Fitting]
fitter = "ideal"

[Fitting.model]
type = "biexp"

[Fitting.solver]
type     = "curvefit"
max_iter = 250
tol      = 1e-8

[Fitting.solver.p0]
S0 = 1000.0
f1 = 0.2
D1 = 0.001
D2 = 0.02

[Fitting.solver.bounds]
S0 = [1.0,   5000.0]
f1 = [0.01,  0.99]
D1 = [1e-5,  0.003]
D2 = [0.003, 0.3]

[Fitting.ideal]
dim_steps              = [[16, 16], [32, 32], [64, 64], [128, 128]]
ideal_dims             = 2
segmentation_threshold = 0.025
downsampling_method    = "block_average"
upsampling_method      = "cubic"

[Fitting.ideal.step_tol]
S0 = 0.5
f1 = 0.2
D1 = 0.2
D2 = 0.2
```

**Python**

```python
from pyneapple.io import load_config, load_dwi_nifti, load_bvalues

image, nifti_ref = load_dwi_nifti("dwi.nii.gz")
bvalues          = load_bvalues("dwi.bval")

fitter = load_config("ideal_biexp.toml").build_fitter()
fitter.fit(xdata=bvalues, image=image)

params = fitter.get_fitted_params()
# {"S0": ndarray, "f1": ndarray, "D1": ndarray, "D2": ndarray}
```

---

### Segmented (two-step bi-exponential)

**TOML**

```toml
[Fitting]
fitter = "segmented"

[Fitting.model]
type        = "biexp"
fit_reduced = true

[Fitting.solver]
type     = "curvefit"
max_iter = 500
tol      = 1e-8

[Fitting.solver.p0]
f1 = 0.2
D1 = 0.01
D2 = 0.001

[Fitting.solver.bounds]
f1 = [0.0, 1.0]
D1 = [1e-4, 0.1]
D2 = [1e-5, 0.01]

[Fitting.segmented]
step1_bvalue_range = [200, null]
fixed_from_step1   = ["D"]
param_mapping      = {D = "D2"}

[Fitting.segmented.step1.model]
type = "monoexp"

[Fitting.segmented.step1.solver]
type     = "curvefit"
max_iter = 250
tol      = 1e-8

[Fitting.segmented.step1.solver.p0]
S0 = 1.0
D  = 0.001

[Fitting.segmented.step1.solver.bounds]
S0 = [0.01, 5.0]
D  = [1e-5, 0.1]
```

**Python**

```python
from pyneapple.io import load_config, load_dwi_nifti, load_bvalues

image, nifti_ref = load_dwi_nifti("dwi.nii.gz")
bvalues          = load_bvalues("dwi.bval")

fitter = load_config("segmented_biexp.toml").build_fitter()
fitter.fit(xdata=bvalues, image=image)

params = fitter.get_fitted_params()
# {"f1": ndarray, "D1": ndarray, "D2": ndarray}
step1 = fitter.step1_params_
# {"S0": ndarray, "D": ndarray}
```

---

### NNLS

**TOML**

```toml
[Fitting]
fitter = "pixelwise"

[Fitting.model]
type    = "nnls"
d_range = [0.0008, 0.5]
n_bins  = 250

[Fitting.solver]
type      = "nnls"
reg_order = 2
mu        = 0.02
max_iter  = 250
tol       = 1e-8
```

**Python**

```python
from pyneapple.io import load_config, load_dwi_nifti, load_bvalues

image, nifti_ref = load_dwi_nifti("dwi.nii.gz")
bvalues          = load_bvalues("dwi.bval")

fitter = load_config("nnls.toml").build_fitter()
fitter.fit(xdata=bvalues, image=image)

params = fitter.get_fitted_params()
# {"coefficients": ndarray shape (X, Y, Z, n_bins)}
```

---

## See also

- [Configuration reference](configuration.md) — every TOML key explained
- [Python API](python-api.md) — construct models and solvers directly in Python
- [IDEAL Fitting](ideal-fitting.md) — `[Fitting.ideal]` keys in detail
- [Segmented Fitting](segmented-fitting.md) — two-step fitting workflow
