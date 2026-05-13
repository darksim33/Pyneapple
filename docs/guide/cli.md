# CLI reference — `pyneapple`

> **TL;DR** — `pyneapple` is the single entry point for all fitting modes. Covers the command synopsis, every shared flag and its default, per-command differences (`pixelwise`, `segmented`, `ideal`, `info`), output file naming, exit codes, and runnable examples.

---

## Synopsis

```bash
pyneapple <command> [options]
```

Run `pyneapple --help` or `pyneapple <command> --help` for the full option list.

---

## Commands

| Command | Description |
|---|---|
| `pixelwise` | Fit each voxel independently |
| `segmented` | Fit the mean signal per labelled ROI (`--seg` required) |
| `ideal` | IDEAL iterative multi-resolution fitting |
| `info` | Print version and available models / solvers / fitters |

---

## Shared options

All fitting commands (`pixelwise`, `segmented`, `ideal`) accept the same core flags:

| Flag | Short | Required | Default | Description |
|---|---|---|---|---|
| `--image` | `-i` | yes | — | 4-D DWI NIfTI image (`.nii` / `.nii.gz`) |
| `--bval` | `-b` | yes | — | B-value file — one value per line or space-separated |
| `--config` | `-c` | yes | — | TOML fitting configuration file |
| `--seg` | `-s` | command-dependent | `None` | Segmentation mask NIfTI (required for `segmented`) |
| `--output` | `-o` | no | image directory | Output directory for parameter maps |
| `--fixed` | `-f` | no | — | Fix a parameter to a NIfTI map: `NAME:PATH` (repeatable) |
| `--diagnostics` | `-d` | no | off | Write `<stem>_diagnostics.h5` alongside the NIfTI maps (see [Output files](#output-files)) |
| `--verbose` | `-v` | no | off | Enable DEBUG-level logging |

### Per-command differences

- **`pixelwise`** — `--seg` is optional; when supplied, only non-zero voxels are fitted.
- **`segmented`** — `--seg` is **required**; the mean signal of each labelled ROI is fitted.
- **`ideal`** — `--seg` is optional; the TOML config must include a `[Fitting.ideal]` section.

---

## Output files

The image stem is derived by stripping `.nii` or `.nii.gz` from the input filename.
One compressed NIfTI is written per fitted parameter:

```
dwi.nii.gz → dwi_S0.nii.gz, dwi_D.nii.gz
subject01.nii → subject01_S0.nii.gz, subject01_D.nii.gz
```

For NNLS fits the single output `<stem>_coefficients.nii.gz` is a 4-D volume
of shape `(X, Y, Z, n_bins)`.

When `--diagnostics` / `-d` is supplied, one additional HDF5 file is written:

```
dwi.nii.gz → dwi_diagnostics.h5
```

The diagnostics file contains per-voxel convergence flags, R², residuals, parameter covariance matrices, and iteration counts — useful for quality-control and troubleshooting.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Configuration or fitting error (bad config, unknown model type, …) |
| `2` | Input file not found (or required option missing) |

---

## Examples

Fit a mono-exponential model, write results to `./results/`:

```bash
pyneapple pixelwise \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config monoexp_config.toml \
    --output ./results
```

Fit only a masked region with verbose logging:

```bash
pyneapple pixelwise \
    -i subject01.nii.gz \
    -b subject01.bval \
    -c nnls_config.toml \
    -s brain_mask.nii.gz \
    -o ./results \
    -v
```

Segmentation-wise fitting (mean signal per ROI):

```bash
pyneapple segmented \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config biexp_seg.toml \
    --seg    roi_mask.nii.gz \
    --output ./results
```

IDEAL fitting (requires `[Fitting.ideal]` section in config):

```bash
pyneapple ideal \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config ideal_biexp.toml \
    --output ./results
```

Fix `T1` to a pre-computed map during fitting:

```bash
pyneapple pixelwise \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config monoexp_config.toml \
    --fixed  T1:t1_map.nii.gz \
    --output ./results
```

`--fixed` can be repeated to fix multiple parameters:

```bash
pyneapple pixelwise \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config biexp_config.toml \
    --fixed  D2:slow_diff.nii.gz \
    --fixed  S0:s0_map.nii.gz \
    --output ./results
```

Print installed version and registered components:

```bash
pyneapple info
```
