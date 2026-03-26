# Quickstart

> **TL;DR** — Install Pyneapple, prepare a NIfTI image and b-value file, write a two-section TOML config, and run `pyneapple-pixelwise` to produce one parameter map per fitted parameter. Covers installation, required inputs, config format with a worked mono-exponential example, the CLI command, and suggested next steps.

---

## 1. Install

```bash
uv pip install pyneapple
```

Requires Python ≥ 3.12. See [installation details in the README](../../README.md#installation) for the development setup.

---

## 2. Prepare your inputs

You need three files:

| File | Description |
|---|---|
| `dwi.nii.gz` | 4-D DWI NIfTI image `(X, Y, Z, N_b)` |
| `dwi.bval` | B-values, one per line (plain text) |
| `config.toml` | Fitting configuration (see below) |

---

## 3. Write a config

Mono-exponential fit with curve fitting:

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

Ready-to-use examples live in [`examples/`](../../examples/).

> **Tip** — You can hold individual model parameters constant during fitting (e.g. a pre-computed T1 map). Add a `[Fitting.model.fixed_params]` table to the config for scalar values, or use the `--fixed` CLI flag for per-pixel NIfTI maps. See [Fixed parameters](python-api.md#fixed-parameters) for details.

---

## 4. Run

```bash
pyneapple-pixelwise \
    --image  dwi.nii.gz \
    --bval   dwi.bval \
    --config config.toml \
    --output ./results
```

Outputs are written as `<image_stem>_<parameter>.nii.gz` — e.g. `dwi_S0.nii.gz` and `dwi_D.nii.gz`.

---

## Next steps

- Full list of config keys → [Configuration](configuration.md)
- All CLI flags → [CLI reference](cli.md)
- Model details and equations → [Models](../concepts/models.md)
- Architecture overview → [Architecture](../concepts/architecture.md)
