# Naming conventions

| variable   | assignment                                   |
|------------|----------------------------------------------|
| `d`        | diffusion coefficients                       |
| `f`        | volume fractions                             |
| `b_values` | b-values                                     |
| `S0`       | Signal at b = 0                              |
| `d_range`  | diffusion fitting range                      |
| `bins`     | log spaced values inside `d_range`           |
| `n_bins`   | number of `bins`                             |
| `x0`       | starting values for NLLS                     |
| `spectrum` | spectrum                                     |
| `img`      | MRI image                                    |
| `seg`      | segmentation/ROI of `img`                    |
| `model`    | one of the following: NLLS, IDEAL, NNLS, ... |
| `n_pools`  | number of cpu kernels for multi-processing   |
