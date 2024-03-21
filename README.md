# PyNeappleUI

> "Pineapple is one of my favorite fruits. Just not on pizza."  
> _- T. Thiel, Co-Founder and CEO of Pyneapple Global Ltd._

> "When life gives you pineapples, make _tropical_ lemonade!"\
> _- J. Jasse, Co-Founder and President of Pyneapple Global Ltd._

## Why PyNeapple?

PyNeapple is an advanced tool for analysing multi-exponential signal data in MR DWI images. It is able to apply a
variety of different fitting algorithms (NLLS, IDEAL, NNLS, ...) to the measured diffusion data and to compare
multi-exponential fitting methods. Thereby it can determine the total number of components contributing to the
corresponding multi-exponential signal fitting approaches and analyses the results by calculating the corresponding
diffusion parameters. Fitting can be customised to be performed on a pixel by pixel or segmentation-wise basis.

> "A juicy tropical delight, an exquisite golden nectar of the tropics. Just as this code is exquisite and golden nectar
> of pure genius."  
> _- Steve Jobs, Former chairman and CEO of another, similarly successful fruit-named company_


## I love their delicious juice, but how does PyNeapple work?

After defining an image and segmentation file using the specified Nii class

```python
img = utils.Nii(Path(r"image.nii"))
seg = utils.NiiSeg(Path(r"segmentation.nii"))
```

a fitting object is created by specifying the desired model, e.g. the NNLS model, and passing the image and
segmentation (and optionally loading a json file containing the desired fitting parameters):

```python
data = FitData(model="NNLS", img=image, seg=segmentation)

# Optional loading of fitting parameters:
data.fit_params.load_from_json(r"fitting_parameters_NNLS.json")
```

```FitData``` then initialises a [fitting model](#the-model-class) with said model properties, other (partially model 
specific) fitting parameters such as b-values, maximum iterations, number of CPUs and [many more](#the-json-not-derulo) 
and a placeholder for future results. If no fitting parameters are provided, the ```FitData``` class will initialise default
parameters dependent on the chosen fitting model.

Fitting can either be done pixel-wise or for whole segmentation images:

```python
data.fit_pixel_wise()
data.fit_segmentation_wise(multi_threading=True)

# Optional applying of the AUC constraint:
d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(data.fit_results)
```

It is carried out by the ```fit``` module, which stores the results in the nested ```Results``` class. This object then
contains all evaluated diffusion parameters such as d- and f-values and results for S0 and T1, if applicable. Optionally,
a boolean can be passed to enable the multi-threading feature of PyNeapple (set ```True``` by default). After fitting, an
AUC constraint can be applied to the results, for the NNLS_AUC fitting approach or for general AUC smoothing of the
acquired data.

Following the fitting procedure, the results can be saved to an Excel or NifTi file. Additionally, a heatmap
of the fitted pixels or ROIs can also be generated and saved.

> "In a world full of apples, be a pineapple."\
> _- Sir Isaac Newton, Apple (tree) enthusiast and revolutionary of modern physics_

___
## Deeper tropical fruit lore
### The Model class

 By creating a model using the ```Model``` class

 ```python
class Model:
    class NNLS(object):
        @staticmethod
        def fit(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int | None):
            fit, _ = nnls(basis, signal, maxiter=max_iter)

            return idx, fit
 ```

 it returns the model-specific fit of the signal and passes it to the corresponding parameter class (in this
 case ```NNLSParams```) which adds default model-specific parameters (e.g. number of bins, maximum iterations,
 diffusion range) and allows manipulation and output of the different fitting characteristics and parameters.

### The json (not Derulo)

 By writing all relevant fitting parameters into a json file, correct fitting of the image data and storage of your 
 fitting parameters is ensured. Due to strong dependencies on initial fitting parameters in some of the implemented 
 approaches, it is strongly recommended to use a specified json file with an adapted parameter set for each model 
 (and image region). The json file can contain the following basic fitting parameters:
 
| name              | description                                | value                                                        |
|-------------------|--------------------------------------------|--------------------------------------------------------------|
| ```Class```       | corresponding parameter class              | "IVIMParams", "IDEALParams", "NNLSParams", "NNLSregCVParams" |
| ```b-values```    |                                            | list of ints                                                 |
| ```fit-area```    | fitting mode                               | "pixel" or "segmentation"                                    |
| ```max_iter```    | maximum iterations                         | int                                                          |
| ```n_pools```     | number of pools (CPUs) for multi-threading | int                                                          |
| ```d_range```     | fitting range                              | list containing min and max value                            |
| ```scale_image``` | scaling image mode                         | On ("S/S0") or off (False)                                   | 
| ```bins```        | diffusion coefficients used for fitting    | list of doubles                                              |

 Additionally, model-specific parameters can be included. These vary from model to model, an overview about model-own
 parameters is given below.
 For further information about the included and possible parameters passed by the json file, as well as
 detailed information about shape and datatype, please refer to the default parameter files in [resources/fitting](./resources/fitting)
 
| name                         | description                                                                                 | value                              |
|:-----------------------------|:--------------------------------------------------------------------------------------------|:-----------------------------------|
| **IVIM specific**            |                                                                                             |                                    |
| ```n_components```           | number of underlying diffusion components                                                   | int                                |
| ```boundaries```             | dict of lists containing initial staring and boundary parameters (as well as ```d_range```) |                                    |
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

---
v0.7.1
