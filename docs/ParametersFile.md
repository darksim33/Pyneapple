# Parameters File

Here we will explain the parameter file entry by entry based on the final [toml file](../examples/parameters/params_biexp.toml) for the biexponential ivim fitting.

``` toml
[General]
Class = "IVIMParams"  # This identifier is used in the fitting to create a matching parameters class and is the base of the file
description = "Biexponential fitting parameters for kidney data."  #  (optional) description of the used file
fit_type = "multi"  # defines wether the fitting should use multi threating, run on a single core or utilize a gpu
n_pools = 4  # (optional) number of pools used for the multi approach 
max_iter = 250  # number of iterations the solver uses to fit
b_values = [  # x-axis data
    0,
    10,
    20,
    30,
    40,
    50,
    70,
    100,
    150,
    200,
    250,
    350,
    450,
    550,
    650,
    750
]

[Model]  # here are the model specific parameters descripte in more detail
model = "biexp"  # the desired model to fit 
fit_reduced = false  # (optional) the reduced model is used to reduce the complexity of the data by assuming the data is normalized
fit_s0 = false  # (optional) instead of fittig a*exp(-x_data*b) + ... S0 * (a*exp(-x_data*b) + ... ) models are used 
fit_t1 = false  # (optional) additional term for t1 fitting of steam datra
mixing_time = 20  # (optional) needed for t1 fitting of steam data 

[boundaries.D]  # necessary boundary values for the model: f1 * exp(-b_values*D1) + f2 * exp(-b_values*D2)
1 = [0.001, 0.0007, 0.05]
2 = [0.02, 0.003, 0.3]

[boundaries.f]
1 = [85, 10, 500]
2 = [20, 1, 100]
```