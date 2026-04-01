# CLI reference — `pyneapple-pixelwise`

> **TL;DR** — `pyneapple-pixelwise` takes an image, a b-value file, and a TOML config and writes one `<stem>_<param>.nii.gz` per fitted parameter. Optionally fix parameters to pre-computed NIfTI maps with `--fixed`. Covers the CLI synopsis, all argument flags and their defaults, output file naming conventions, exit codes, and usage examples.

---

## Synopsis

```bash
pyneapple-pixelwise --image <PATH> --bval <PATH> --config <PATH> [options]
```

---

## Arguments

| Argument | Short | Required | Description |
|---|---|---|---|
| `--image` | `-i` | yes | 4-D DWI NIfTI image (`.nii` / `.nii.gz`) |
| `--bval` | `-b` | yes | B-value file — one value per line or space-separated |
| `--config` | `-c` | yes | TOML fitting configuration file |
| `--seg` | `-s` | no | Segmentation mask NIfTI — only non-zero voxels are fitted |
| `--output` | `-o` | no | Output directory (defaults to the directory containing the image) |
| `--fixed` | `-f` | no | Fix a parameter to a NIfTI map: `NAME:PATH` (repeatable) |
| `--verbose` | `-v` | no | Enable DEBUG-level logging |

---

## Output files

The image stem is derived by stripping `.nii` or `.nii.gz` from the input filename:

```
dwi.nii.gz → dwi_S0.nii.gz, dwi_D.nii.gz
subject01.nii → subject01_S0.nii.gz, subject01_D.nii.gz
```

For NNLS fits the single output `<stem>_coefficients.nii.gz` is a 4-D volume of shape `(X, Y, Z, n_bins)`.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Configuration or fitting error (bad config, unknown model type, …) |
| `2` | Input file not found |
| `3` | Fixed-parameter map spatial shape mismatch |

---

## Examples

Fit a mono-exponential model, write results to `./results/`:

```bash
pyneapple-pixelwise \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config monoexp_config.toml \
    --output ./results
```

Fit only a masked region with verbose logging:

```bash
pyneapple-pixelwise \
    -i subject01.nii.gz \
    -b subject01.bval \
    -c nnls_config.toml \
    -s brain_mask.nii.gz \
    -o ./results \
    -v
```

Fix `T1` to a pre-computed map during fitting:

```bash
pyneapple-pixelwise \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config monoexp_config.toml \
    --fixed  T1:t1_map.nii.gz \
    --output ./results
```

The `--fixed` flag can be repeated to fix multiple parameters:

```bash
pyneapple-pixelwise \
    --image  subject01.nii.gz \
    --bval   subject01.bval \
    --config biexp_config.toml \
    --fixed  D2:slow_diff.nii.gz \
    --fixed  S0:s0_map.nii.gz \
    --output ./results
```
