# GPU Acceleration

> **TL;DR** — Pyneapple supports GPU-accelerated curve fitting through the optional [pyneapple-gpufit](https://github.com/darksim33/pyneapple-gpufit) plugin. Install it to replace the CPU solver with a batched CUDA implementation powered by [Gpufit](https://github.com/darksim33/GPUfit).

---

## Overview

By default, Pyneapple fits diffusion MRI models on the CPU using SciPy's `curve_fit`. For large datasets, the `pyneapple-gpufit` plugin offloads fitting to an NVIDIA GPU: all pixels are submitted in a single batched call, giving significant throughput improvements without changing the Pyneapple API.

## Installation

```bash
# pip
pip install git+https://github.com/darksim33/pyneapple-gpufit.git

# uv
uv add git+https://github.com/darksim33/pyneapple-gpufit
```

**Requirements:** Python ≥ 3.12, an NVIDIA GPU, and a CUDA-compatible driver. The native library is bundled — no CUDA toolkit installation is needed.

## Usage

Switch the solver by setting `type = "gpufit_curvefit"` in your TOML config:

```toml
[Fitting.solver]
type = "gpufit_curvefit"

[Fitting.solver.p0]
f1 = 0.2
D1 = 0.010
D2 = 0.001

[Fitting.solver.bounds]
f1 = [0.0,  1.0 ]
D1 = [1e-4, 0.1 ]
D2 = [1e-5, 0.01]
```

No other changes to your workflow are required.

## Supported models

| Model | `fit_reduced` | Solver key |
|---|---|---|
| `MonoExpModel` | — | `gpufit_curvefit` |
| `BiExpModel` | `True` (default) | `gpufit_curvefit` |
| `BiExpModel` | `False` | `gpufit_curvefit` |
| `TriExpModel` | `True` (default) | `gpufit_curvefit` |
| `TriExpModel` | `False` | `gpufit_curvefit` |

Models with T1 correction (`fit_t1=True`) or `fit_s0=True` fall back to the CPU `CurveFitSolver`.

## Citation

If you use GPU-accelerated fitting in published work, cite the Gpufit paper:

> Przybylski, A., Throm, B., Kaderali, L. & Grüll, H.\
> **Gpufit: An open-source toolkit for GPU-accelerated curve fitting.**\
> *Scientific Reports* **7**, 15722 (2017).\
> <https://doi.org/10.1038/s41598-017-15313-9>

## Further reading

- [pyneapple-gpufit repository](https://github.com/darksim33/pyneapple-gpufit) — source, issue tracker, and full API reference
- [API Reference](api-reference.md) — solver constructor arguments, `get_params()` / `get_diagnostics()` keys, and supported model configurations
