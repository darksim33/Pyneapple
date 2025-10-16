# Fitting Parameters

By writing all relevant fitting parameters into a *json* or *toml* [(for more details see)](Fitting.md) file, correct fitting of the image data and storage of your
fitting parameters is ensured. Due to strong dependencies on initial fitting parameters in some of the implemented
approaches, it is strongly recommended to use a specified file with an adapted parameter set for each model
(and image region). The file is structured in three main parts, *General*, *Model* and *Boundaries*. For a full entry by entry explanation based on the final file [see](ParametersFile.md)

# General Parameters

| name              | description                                 | value                                                              |
|:------------------|:--------------------------------------------|:-------------------------------------------------------------------|
| ``Class``         | corresponding parameter class               | "IVIMParams",  "IVIMSegmentedParams", "NNLSParams", "NNLSCVParams" |
| ``description``   | Description of the fit                      | str                                                                |
| ``fit_type``      | fitting approach used                       | "single", "multi", "gpu"                                           |
| ``max_iter``      | maximum iterations                          | int                                                                |
| ``n_pools``       | number of pools (CPUs) (multi only)         | int                                                                |
| ``fit_tolerance`` | tolerance for convergence check  (gpu only) | float                                                              |
| ``b-values``      | x_axis data                                 | list of ints                                                       |


Example toml code:

``` toml
[General]
Class = "IVIMParams"
description = "IVIM fitting parameters for DWI data"
fit_type = "multi"
max_iter = 250
n_pools = 4
b_values = [0, 50, 100, 200, 400]
```

# Model Specific parameters

Every model has ``model`` string attribute defining the fitting model used. The available vary between the different
parameter classes. The model specific parameters are defined in the *Model* section of the parameter file.

## IVIM

The available models are:
- "monoexp" - mono-exponential fitting
- "biexp" - bi-exponential fitting
- "triexp" - tri-exponential fitting

For the basic IVIM fitting, the model-specific parameters are defined in the *Model* section of the parameter file.
These are as follows:

| name             | description                                       | value      |
|:-----------------|:--------------------------------------------------|:-----------|
| ``fit_reduced``  | whether reduced fitting should be used            | bool       |
| ``fit_S0``       | wether S0 should be calculated instead directly   | bool       |
| ``fit_t1``       | whether T1 should be calculated in the first step | bool       |
| ``mixing_time``  | mixing time for IVIM fitting                      | int, float |

Example toml code:

``` toml
[Model]
model = "biexp"
fit_reduced = false
fit_S0 = false
fit_t1 = true
mixing_time = 20
```

### Segmeted IVIM

The segmented IVIM fitting uses the same models as the normal IVIM fitting but adds some additional parameters To the 
*General* section. The model specific parameters are:

| name                 | description                                                   | value      |
|:---------------------|:--------------------------------------------------------------|:-----------|
| ``fixed_component``  | name of component to fix (e.g. "D_1")                         | str        |
| ``fixed_t1``         | whether T1 should be calculated in the first step             | bool       |
| ``reduced_b_values`` | can be *false* or a list of *b_values* for initial fitting    | bool, list |

Example toml code:

``` toml
[General]
# ... other general parameters ...
fixed_component = "D_1"
fixed_t1 = true
reduced_b_values = [0, 50, 100, 200, 400]
```

## NNLS

The NNLS Model adds some additional parameters to the *Model* section. The model specific parameters are:

| name            | description                               | value      |
|:----------------|:------------------------------------------|:-----------|
| ``reg_order``   | regularization order (0-3 or "CV")        | int, str   |
| ``mu``          | regularization factor (0 - 3 only)        | float      |
| ``tol``         | tolerance for convergence check (CV only) | float      |

Example toml code:

``` toml
[Model]
reg_order = 2
mu = 0.02
tol = 1e-6
```

# Boundaries

The boundaries value ist a dictionary holding sub dictionaries for each component. For IVIM there are generally three 
options. The first is the classic model with non-relative fractions and without S0 fitting. The second is the *S0* 
approach with relative fractions and *S0* and the third is the reduced bi-exponential fitting, which assumes the data to
be normalized.

## IVIM
For the IVIM fitting each parameter has its own boundaries. The boundaries are defined in the *Boundaries* section of 
the parameter file. Additionally if the *fit_t1* parameter is set to *true*, the *T1* boundaries are defined as well.
Example toml code for reduced bi-exponential:

``` toml
[Boundaries]
# Boundaries for reduced bi-exponential fitting
[Boundaries.D.1]
# Array with [x0, lb, ub]
# [0.02, 0.003, 0.3]
0 = 0.02
1 = 0.003
2 = 0.3

[Boundaries.D.2]
# [0.001, 0.0007, 0.05]
0 = 0.001
1 = 0.0007
2 = 0.05

[Boundaries.f.1]
# [85, 10, 500]
0 = 85
1 = 10
2 = 500

[Boundaries.f.2]
# [20, 1, 100]
0 = 20
1 = 1
2 = 100
```

## NNLS

Since the NNLS method takes a different approach in calculating the underlying diffusion parameters a different set of boundaries and additional parameters is needed.

Boundaries:

| name        | description                                  | type               |
|-------------|----------------------------------------------|--------------------|
| ``d_range`` | diffusion value range for NNLS fitting       | list(float, float) |
| ``n_bins``  | number of exponential terms used for fitting | int                |

---
