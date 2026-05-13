# Configuration reference

> **TL;DR** — A Pyneapple config is a TOML file with a `[Fitting]` root containing `[Fitting.model]` and `[Fitting.solver]` sub-sections. Covers the top-level fitter key, all model types and their config keys (including modes and T1 correction), both solver types (`curvefit` and `nnls`) with their full key references, multi-threading options, and complete worked examples.

---

## Top-level keys

| Key | Type | Required | Description |
|---|---|---|---|
| `fitter` | string | yes | Fitter type — see table below |

### Supported fitter types

| `fitter` | Class | Description |
|---|---|---|
| `"pixelwise"` | `PixelWiseFitter` | Fits each voxel independently |
| `"ideal"` | `IDEALFitter` | Iterative multi-resolution fitting — requires `[Fitting.ideal]` section |
| `"segmented"` | `SegmentedFitter` | Two-step fitting: monoexp on high b-values, then full model with fixed parameter |
| `"segmentationwise"` | `SegmentationWiseFitter` | Fits one set of parameters per segmentation region |

```toml
[Fitting]
fitter = "pixelwise"
```

---

## `[Fitting.model]`

| Key | Type | Required | Description |
|---|---|---|---|
| `type` | string | yes | Model type — see table below |
| `d_range` | [float, float] | NNLS only | Diffusion coefficient range `[D_min, D_max]` in mm²/s |
| `n_bins` | int | NNLS only | Number of logarithmically spaced bins |
| `fit_reduced` | bool | biexp / triexp | Constrain last fraction (default `true`) |
| `fit_s0` | bool | biexp / triexp | Add S0 amplitude parameter (requires `fit_reduced = true`) |
| `fit_t1` | bool | — | Enable standard T1 relaxation fitting |
| `fit_t1_steam` | bool | — | Enable STEAM T1 fitting (implies `fit_t1 = true`) |
| `repetition_time` | float | when `fit_t1 = true` | Repetition time TR in ms |
| `mixing_time` | float | when `fit_t1_steam = true` | Mixing time TM in ms |
| `fixed_params` | table | no | Parameters held constant during fitting — `{name = value}` |

### Supported model types

| `type` | Model | Fitted parameters (default mode) |
|---|---|---|
| `monoexp` | Mono-exponential | `S0`, `D` |
| `biexp` | Bi-exponential | `f1`, `D1`, `D2` |
| `triexp` | Tri-exponential | `f1`, `D1`, `f2`, `D2`, `D3` |
| `nnls` | NNLS distribution | `coefficients` (length `n_bins`) |

### Fixed parameters

Fix one or more model parameters to constant scalar values with a `[Fitting.model.fixed_params]` sub-table. Fixed parameters are excluded from optimization and held at the specified values during fitting.

```toml
[Fitting.model]
type = "biexp"

[Fitting.model.fixed_params]
D2 = 0.003
```

In this example the slow diffusion coefficient `D2` is held at `0.003` mm²/s and only `f1` and `D1` are fitted. You can fix any parameter listed in the model's parameter table except for NNLS coefficients.

Bi- and tri-exponential models default to reduced mode (`fit_reduced = true`), which constrains the last fraction to preserve the sum-to-one constraint. Pass `fit_s0 = true` to add an `S0` amplitude parameter, or `fit_reduced = false` for fully independent fractions. See [Model modes](../concepts/models.md#model-modes).

---

## `[Fitting.solver]`

### Common keys

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | — | `"curvefit"`, `"constrained_curvefit"`, or `"nnls"` |
| `max_iter` | int | 250 | Maximum solver iterations |
| `tol` | float | 1e-8 | Convergence tolerance |

### `curvefit`-specific

| Key | Type | Default | Description |
|---|---|---|---|
| `multi_threading` | bool | false | Enable parallel voxel fitting |
| `n_pools` | int | — | Number of worker processes (`-1` = all cores) |
| `[Fitting.solver.p0]` | table | — | Initial parameter guesses, keyed by parameter name |
| `[Fitting.solver.bounds]` | table | — | `[lower, upper]` bounds per parameter |

### `constrained_curvefit`-specific

Uses `scipy.optimize.minimize` with the SLSQP method, which supports both box bounds and the inequality constraint `sum(f_i) <= 1`. Use this solver instead of `"curvefit"` when you want to prevent volume fractions from summing above one.

> **Requires** `fit_reduced = true` on the model. Passing `fit_reduced = false` raises a `ValueError` at config load time because the hard fraction constraint is only physically meaningful for normalised (reduced) signals.

| Key | Type | Default | Description |
|---|---|---|---|
| `multi_threading` | bool | false | Enable parallel voxel fitting |
| `n_pools` | int | — | Number of worker processes (`-1` = all cores) |
| `fraction_constraint` | bool | true | Enforce `sum(f_i) <= 1` via SLSQP inequality constraint |
| `[Fitting.solver.p0]` | table | — | Initial parameter guesses, keyed by parameter name |
| `[Fitting.solver.bounds]` | table | — | `[lower, upper]` bounds per parameter |

### `nnls`-specific

| Key | Type | Default | Description |
|---|---|---|---|
| `multi_threading` | bool | false | Enable parallel voxel fitting |
| `n_pools` | int | — | Number of worker processes (`-1` = all cores) |
| `reg_order` | int | 0 | Regularisation order — `0` = none, `1` = first diff, `2` = second diff, `3` = curvature |
| `mu` | float | 0.02 | Regularisation strength |

### Multi-threading

Both solver types support parallel voxel fitting via [joblib](https://joblib.readthedocs.io):

| `n_pools` value | Behaviour |
|---|---|
| key absent / `1` | Single-process (default) — lowest overhead, best for debugging |
| `-1` | Use all available CPU cores |
| `N` (positive int) | Use exactly N worker processes |

Enable for datasets larger than ~500 voxels. For NNLS fits, parallelism is over voxels — the bin-grid computation is not parallelised.

```toml
[Fitting.solver]
type            = "curvefit"
multi_threading = true
n_pools         = -1   # all cores
```

---

## Segmented fitter (`fitter = "segmented"`)

When `fitter = "segmented"`, Step 2 uses the top-level `[Fitting.model]` and `[Fitting.solver]` sections. Step 1 is configured inside `[Fitting.segmented]`.

| Key in `[Fitting.segmented]` | Type | Required | Description |
|---|---|---|---|
| `step1_bvalue_range` | `[lo, hi]` | yes | B-value range for Step 1 — `null` = open end, e.g. `[200, null]` for b >= 200 |
| `fixed_from_step1` | array of strings | no | Step 1 parameter names to pass as per-pixel fixed params in Step 2 |
| `param_mapping` | table | no | Maps Step 1 names to Step 2 names, e.g. `{D = "D2"}` |
| `[Fitting.segmented.step1.model]` | table | yes | Step 1 model config (same keys as `[Fitting.model]`) |
| `[Fitting.segmented.step1.solver]` | table | yes | Step 1 solver config (same keys as `[Fitting.solver]` plus `p0` and `bounds`) |

### Example

```toml
[Fitting]
fitter = "segmented"

# Step 2: bi-exponential on all b-values
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

# Step 1: mono-exponential on high b-values
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

Step 1 defaults to model type `"monoexp"` and solver type `"curvefit"` when the sub-sections are omitted.

---

## Full examples

### Mono-exponential

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

### NNLS

```toml
[Fitting]
fitter = "pixelwise"

[Fitting.model]
type    = "nnls"
d_range = [0.0008, 0.5]
n_bins  = 250

[Fitting.solver]
type            = "nnls"
reg_order       = 2
mu              = 0.02
max_iter        = 250
tol             = 1e-8
multi_threading = true
n_pools         = 4
```

### Bi-exponential with constrained fractions

Uses `"constrained_curvefit"` to enforce `f1 + f2 <= 1` during fitting. `fit_reduced = true` is required.

```toml
[Fitting]
fitter = "pixelwise"

[Fitting.model]
type        = "biexp"
fit_reduced = true

[Fitting.solver]
type                 = "constrained_curvefit"
fraction_constraint  = true
max_iter             = 500
tol                  = 1e-8
multi_threading      = true
n_pools              = -1

[Fitting.solver.p0]
f1 = 0.2
D1 = 0.01
D2 = 0.001

[Fitting.solver.bounds]
f1 = [0.0, 1.0]
D1 = [1e-4, 0.1]
D2 = [1e-5, 0.01]
```
