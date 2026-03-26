# Architecture

> **TL;DR** — Pyneapple is built on three independent layers: Model (forward physics), Solver (optimisation), and Fitter (spatial orchestration), each with a single responsibility and no knowledge of the layers above it. Covers the layer overview and data-flow diagram, the two model types (parametric vs distribution), the two solver types (CurveFit vs NNLS), and the five key design rules that govern the architecture.

---

## Layer overview

```
Fitter
  │  Iterates over voxels, manages spatial indexing
  │
  └─▶ Solver
        │  Runs the numerical optimisation
        │
        └─▶ Model
              Pure forward function — no state, no fitting
```

| Layer | Responsibility | Must not |
|---|---|---|
| `Model` | Compute `forward(xdata, *params)` | Know about fitting or spatial data |
| `Solver` | Minimise residual, own fitted state (`params_`) | Know about image shape or voxels |
| `Fitter` | Extract pixels, dispatch to solver, reconstruct maps | Implement physics or optimisation |

---

## Model types

### `ParametricModel`

Implements a closed-form forward equation:

```python
model.forward(b_values, S0, D)  # → signal array
```

### `DistributionModel`

Owns `bins` (log-spaced D grid) and `get_basis(xdata)` (exponential decay matrix). The solver calls these to build the regularised least-squares system:

```python
model.bins          # shape (n_bins,)
model.get_basis(b)  # shape (n_measurements, n_bins)
model.forward(b, *spectrum)  # basis @ spectrum
```

---

## Solver types

| Solver | Model type | Fitted state |
|---|---|---|
| `CurveFitSolver` | `ParametricModel` | `params_` — one scalar per parameter per pixel |
| `NNLSSolver` | `DistributionModel` | `params_["coefficients"]` — shape `(n_pixels, n_bins)` |

Both solvers store diagnostics in `diagnostics_["residual"]`.

---

## Key design rules

1. **Models are stateless.** `forward()` is a pure function. No `fit()`, no stored data.
2. **Solvers own fitted state.** After `solver.fit()`, results live in `solver.params_` (trailing underscore, scikit-learn convention).
3. **Fitters own spatial state.** `fitter.pixel_indices` maps fitted pixels back to `(x, y, z)` coordinates.
4. **Physics lives in the model.** Bin grids, basis matrices, and decay equations belong to the model layer — not the solver.
5. **Configuration lives in TOML.** The `build_fitter()` method on `FittingConfig` is the single point of construction; no object is built by hand in the CLI.
