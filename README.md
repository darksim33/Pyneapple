<p align="center">
  <img src="docs/assets/logo.svg" alt="Pyneapple logo" width="200"/>
</p>

<p align="center">
  <strong>
    Pyneapple - An advanced tool for analysing multi-exponential signals <br> of Diffusion Weighted MR data.
  </strong>
<p>

## Why Pyneapple?

Pyneapple is an advanced tool for analysing multi-exponential signal data in MR DWI images. It is able to apply a
variety of different fitting algorithms (NLLS, NNLS, ...) to the measured diffusion data and to compare
multi-exponential fitting methods. Thereby it can determine the total number of components contributing to the
corresponding multi-exponential signal fitting approaches and analyses the results by calculating the corresponding
diffusion parameters. Fitting can be customised to be performed on a pixel by pixel or segmentation-wise basis.


## Installation

Requires Python ≥ 3.12. [uv](https://docs.astral.sh/uv/) is recommended for fast, reproducible environment management.

```bash
uv pip install pyneapple
```

For development, clone the repository and install with the dev extras:

```bash
git clone https://github.com/darksim33/Pyneapple.git
cd Pyneapple
uv sync --all-groups
```

Install extras for additional features:

```bash
uv pip install pyneapple[excel,plotting,export]
```

<details>
<summary>pip (without uv)</summary>

```bash
pip install pyneapple
# or for development:
pip install -e ".[dev]"
```

</details>

---

## CLI usage

Pyneapple provides four CLI subcommands for different fitting strategies:

| Command | Description |
|---|---|
| `pyneapple-pixelwise` | Fits each voxel independently |
| `pyneapple-segmentationwise` | Fits mean signal per segmentation region |
| `pyneapple-segmented` | Two-step fitting (simple model → complex model) |
| `pyneapple-ideal` | Iterative multi-resolution IDEAL fitting |

### Pixel-wise fitting

Fits each voxel independently and writes one NIfTI parameter map per fitted parameter.

```
pyneapple-pixelwise --image dwi.nii.gz --bval dwi.bval --config config.toml [options]
```

| Argument | Short | Required | Description |
|---|---|---|---|
| `--image` | `-i` | yes | 4-D DWI NIfTI image (`.nii` / `.nii.gz`) |
| `--bval` | `-b` | yes | B-value file, one value per line |
| `--config` | `-c` | yes | TOML fitting configuration file |
| `--seg` | `-s` | no | Segmentation mask — only non-zero voxels are fitted |
| `--output` | `-o` | no | Output directory (defaults to the image directory) |
| `--verbose` | `-v` | no | Enable DEBUG-level logging |

Output files are named `<image_stem>_<parameter>.nii.gz`.  
For NNLS fits the single output `<image_stem>_coefficients.nii.gz` is a 4-D volume of shape `(X, Y, Z, n_bins)`.

### Segmentation-wise fitting

Fits the mean signal of each segmentation region.

```
pyneapple-segmentationwise --image dwi.nii.gz --bval dwi.bval --seg seg.nii.gz --config config.toml [options]
```

Shares the same arguments as pixelwise, plus:
| `--seg` | `-s` | yes | Segmentation mask with integer labels |

### Segmented fitting

Two-step fitting where a simple model (e.g., monoexponential) provides initial parameters for a complex model (e.g., biexponential).

```
pyneapple-segmented --image dwi.nii.gz --bval dwi.bval --seg seg.nii.gz --config config.toml [options]
```

### IDEAL fitting

Iterative multi-resolution fitting for improved convergence.

```
pyneapple-ideal --image dwi.nii.gz --bval dwi.bval --seg seg.nii.gz --config config.toml [options]
```

---

## Configuration file

A minimal mono-exponential example:

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

A minimal NNLS example:

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

Ready-to-use example configs are in [`examples/`](examples/).

### Supported model types

| `type` | Model | Fitted parameters |
|---|---|---|
| `monoexp` | Mono-exponential | `S0`, `D` |
| `biexp` | Bi-exponential | `S0`, `f`, `D_fast`, `D_slow` |
| `triexp` | Tri-exponential | `f1`, `f2`, `f3`, `D1`, `D2`, `D3` |
| `nnls` | NNLS distribution | `coefficients` (length `n_bins`) |
___