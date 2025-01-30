# The Parameter Json

By writing all relevant fitting parameters into a json file, correct fitting of the image data and storage of your
fitting parameters is ensured. Due to strong dependencies on initial fitting parameters in some of the implemented
approaches, it is strongly recommended to use a specified json file with an adapted parameter set for each model
(and image region). The json file can contain the following basic fitting parameters:

| name | description | value                                                      |
|:-------------------|:--------------------------------------------|:-----------------------------------------------------------|
| ```Class```       | corresponding parameter class | "IVIMParams", "IDEALParams", "NNLSParams", "NNLSCVParams"  |
| ```b-values```    | used for imaging | list of ints                                               |
| ```fit-area```    | fitting mode | "pixel" or "segmentation"                                  |
| ```max_iter```    | maximum iterations | int                                                        |
| ```n_pools```     | number of pools (CPUs) for multi-threading | int                                                        |
| ```d_range```     | fitting range | list containing min and max value                          |
| ```scale_image``` | scaling image mode | On ("S/S0") or off (False)                                 |
| ```bins```        | diffusion coefficients used for fitting | list of doubles                                            |

Additionally, model-specific parameters can be included. These vary from model to model, an overview about model-own
parameters is given below.
For further information about the included and possible parameters passed by the json file, as well as
detailed information about shape and datatype, please refer to the default parameter files
in [resources/fitting](./resources/fitting)

| name                         | description                                                                                 | value                              |
|:-----------------------------|:--------------------------------------------------------------------------------------------|:-----------------------------------|
| **IVIM specific**            |                                                                                             |                                    |
| ```n_components```           | number of underlying diffusion components                                                   | int                                |
| ```boundaries```             | dict of lists containing initial staring and boundary parameters (as well as ```d_range```) | dict of lists                      |
| **additional IDEAL params:** |                                                                                             |                                    |
| ```dimension_steps```        | step sizes for interpolation if the image                                                   | [int, int]                         |
| ```tolerance```              | adjustment tolerance of each initial IVIM parameters between steps                          | list of doubles                    |
| ```segmentation_threshold``` | threshold share per quadrant after interpolation step                                       | double                             |
| **NNLS specific**            |                                                                                             |                                    |
| ```reg_order```              | regularisation order                                                                        | "0-3" or "CV" for cross-validation |
| ```mu```                     | regularisation factor                                                                       | double                             |

## Naming conventions

<div align="center">

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

</div>

___
