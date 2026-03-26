# Configuration reference

> **TL;DR** — A Pyneapple config is a TOML file with a `[Fitting]` root containing `[Fitting.model]` and `[Fitting.solver]` sub-sections. Covers the top-level fitter key, all model types and their config keys (including modes and T1 correction), both solver types (`curvefit` and `nnls`) with their full key references, multi-threading options, and complete worked examples.

---

## Top-level keys

```toml
[Fitting]
fitter = "pixelwise"   # required — only "pixelwise" is supported currently
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
| `type` | string | — | `"curvefit"` or `"nnls"` |
| `max_iter` | int | 250 | Maximum solver iterations |
| `tol` | float | 1e-8 | Convergence tolerance |

### `curvefit`-specific

| Key | Type | Default | Description |
|---|---|---|---|
| `multi_threading` | bool | false | Enable parallel voxel fitting |
| `n_pools` | int | — | Number of worker processes (`-1` = all cores) |
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
