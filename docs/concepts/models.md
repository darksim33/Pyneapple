# Models

> **TL;DR** — Pyneapple provides three parametric models (mono-, bi-, tri-exponential) and one distribution model (NNLS), all sharing the same `forward(xdata, *params)` interface. Parametric models support multiple operating modes (reduced, full, with-amplitude) and optional T1 relaxation correction. Covers model equations and parameters, operating modes, T1 correction keys, and the Python API.

---

## Parametric models

### `MonoExpModel`

$$S(b) = S_0 \cdot e^{-b \cdot D}$$

| Parameter | Unit | Description |
|---|---|---|
| `S0` | a.u. | Signal at b = 0 |
| `D` | mm²/s | Apparent diffusion coefficient |

### `BiExpModel`

$$S(b) = f_1 \cdot e^{-b \cdot D_1} + (1 - f_1) \cdot e^{-b \cdot D_2}$$

Default mode (`fit_reduced=True`, `fit_s0=False`):

| Parameter | Unit | Description |
|---|---|---|
| `f1` | — | Fast diffusion fraction |
| `D1` | mm²/s | Fast diffusion coefficient |
| `D2` | mm²/s | Slow diffusion coefficient |

The slow fraction is implicitly $1 - f_1$. For other modes (full fractions, explicit S0) see [Model modes](#model-modes).

### `TriExpModel`

$$S(b) = f_1 e^{-b D_1} + f_2 e^{-b D_2} + f_3 e^{-b D_3}$$

| Parameter | Unit | Description |
|---|---|---|
| `f1`, `f2`, `f3` | a.u. | Component amplitudes |
| `D1`, `D2`, `D3` | mm²/s | Component diffusion coefficients |

---

## Distribution model

### `NNLSModel`

Represents the signal as a sum over a log-spaced D grid:

$$S(b) = \sum_{k=1}^{N} c_k \cdot e^{-b \cdot D_k}$$

Config keys:

| Key | Description |
|---|---|
| `d_range` | `[D_min, D_max]` — extent of the bin grid in mm²/s |
| `n_bins` | Number of logarithmically spaced bins |

Output: a single `coefficients` map of shape `(X, Y, Z, n_bins)`.

Regularisation is controlled by `reg_order` and `mu` in `[Fitting.solver]` — see [Configuration](../guide/configuration.md).

---

## Model modes

Bi- and tri-exponential models support three operating modes set via `[Fitting.model]` keys. The mode changes which parameters are fitted and their names in the output maps.

### BiExp modes

| Mode | TOML keys | Fitted parameters | Equation |
|---|---|---|---|
| **Reduced** (default) | `fit_reduced = true` | `f1`, `D1`, `D2` | $f_1 e^{-bD_1} + (1-f_1)e^{-bD_2}$ |
| **Full** | `fit_reduced = false` | `f1`, `D1`, `f2`, `D2` | $f_1 e^{-bD_1} + f_2 e^{-bD_2}$ |
| **With amplitude** | `fit_reduced = true`, `fit_s0 = true` | `f1`, `D1`, `D2`, `S0` | $S_0(f_1 e^{-bD_1} + (1-f_1)e^{-bD_2})$ |

> [!NOTE]
> `fit_s0 = true` requires `fit_reduced = true`. Combining independent fractions with an explicit S0 is over-parameterised.

```toml
[Fitting.model]
type       = "biexp"
fit_reduced = false   # full mode — independent f1 and f2
```

### TriExp modes

| Mode | TOML keys | Fitted parameters |
|---|---|---|
| **Reduced** (default) | `fit_reduced = true` | `f1`, `D1`, `f2`, `D2`, `D3` |
| **Full** | `fit_reduced = false` | `f1`, `D1`, `f2`, `D2`, `f3`, `D3` |
| **With amplitude** | `fit_reduced = true`, `fit_s0 = true` | `f1`, `D1`, `f2`, `D2`, `D3`, `S0` |

In reduced mode $f_3 = 1 - f_1 - f_2$; in full mode all three fractions are independent.

---

## T1 correction

All parametric models support optional T1 relaxation correction, applied on top of the base forward function.

**Standard (inversion recovery):**

$$S_{T_1}(b) = S(b) \cdot \left(1 - e^{-TR / T_1}\right)$$

**STEAM sequence:**

$$S_\text{STEAM}(b) = S(b) \cdot \left(1 - e^{-TR / T_1}\right) \cdot e^{-TM / T_1}$$

Enabling either mode appends a `T1` parameter to the model's parameter list and adds a `T1` map to the outputs.

| TOML key | Type | Required when | Description |
|---|---|---|---|
| `fit_t1` | bool | — | Enable standard inversion-recovery T1 correction |
| `fit_t1_steam` | bool | — | Enable STEAM T1 correction (also sets `fit_t1 = true`) |
| `repetition_time` | float | `fit_t1 = true` | Repetition time TR in ms |
| `mixing_time` | float | `fit_t1_steam = true` | Mixing time TM in ms |

```toml
[Fitting.model]
type             = "monoexp"
fit_t1           = true
repetition_time  = 3000.0   # TR in ms
```

```toml
[Fitting.model]
type             = "biexp"
fit_t1_steam     = true
repetition_time  = 3000.0   # TR in ms
mixing_time      = 30.0     # TM in ms
```

---

## Python API

All models share the same interface:

```python
from pyneapple import MonoExpModel
import numpy as np

model = MonoExpModel()
b = np.array([0, 100, 500, 1000], dtype=float)

signal = model.forward(b, 1000.0, 0.001)
# array([1000. ,  905.0,  606.5,  368.0])
```

```python
from pyneapple import NNLSModel

model = NNLSModel(d_range=(0.0008, 0.5), n_bins=250)
basis = model.get_basis(b)   # shape (4, 250)
bins  = model.bins            # shape (250,)
```
