# PyNeappleUI

> "Pineapple is one of my favorite fruits. Just not on pizza."  
> _- T. Thiel, Co-Founder and CEO of Pyneapple Global Ltd._

## Why PyNeapple?

PyNeapple is an advanced tool to analyse multi-exponential signal data in MR DWI images. It is able to apply a
variety of different fitting algorithms (NLLS, IDEAL, NNLS, ...) to the measured diffusion data and compare
multi-exponential fitting methods. Thereby it can determine the total number of components contributing to the
corresponding multi-exponential signal fitting approaches and analyses the results by calculating the corresponding
diffusion parameters. Fitting can be customised to be done pixel by pixel or for segmentation-wise.

> "A juicy tropical delight, an exquisite golden nectar of the tropics. Just as this code is exquisite and golden nectar
> of pure genius."  
> _- Steve Jobs, Former chairman and CEO of another, similarly successful fruit company_

## I love their sweet juice, but how does PyNeapple work?

After defining an image and segmentation file using the specified Nii class

```python
img = utils.Nii(Path(r"image.nii"))
seg = utils.NiiSeg(Path(r"segmentation.nii"))
```

a fitting object is created by specifying the desired model, e.g. the NNLS model, and passing the image and
segmentation (and optionally loading a json file containing desired fitting parameters):

```python
data = FitData(model="NNLS", img=image, seg=segmentation)

# Optional loading of fitting parameters:
data.fit_params.load_from_json(r"fitting_parameters_NNLS.json")
```

```FitData``` then initialises a fitting model with said model properties, further fitting parameters (e.g.
b_values, max_iter, fitting specifications, number of CPUs), a placeholder for future results and an option for
different fitting types (pixel- or segmentation-wise). If no fitting parameters are provided, the ```FitData``` class
initialises default parameters dependent on the chosen fitting model.

> ### Deeper tropical fruit lore
>
>By creating a model using the ```Model``` class
>
> ```python
>class Model:
>    class NNLS(object):
>        def fit(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int | None):
>            fit, _ = nnls(basis, signal, maxiter=max_iter)
>
>            return idx, fit
>```
>
>it returns the model-specific fit of the signal and passes it to the corresponding parameter class (in this
> case ```NNLSParams```) which adds default model-specific parameters (e.g. number of bins, maximum iterations,
> diffusion
> range) and allows manipulation as well as output of the different fitting characteristics and parameters.


Fitting can either be done pixel-wise or for whole segmentation images.

```python
data.fit_pixel_wise()
data.fit_segmentation_wise(multi_threading=True)

# Optional applying of the AUC constraint:
d_AUC, f_AUC = data.fit_params.apply_AUC_to_results(data.fit_results)
```

Fitting is carried out by the ```fit``` module, saving the results into the nested ```Results``` class. This object then
contains all evaluated diffusion parameters like d- and f-values and results for S0 and T1, if applicable. Optionally,
a boolean can be passed, enabling the multi-threading feature of PyNeapple (set ```True``` by default). After fitting an
AUC constraint can be applied to the results, for the NNLS_AUC fitting approach or for general AUC smoothing of the
acquired data.

After the fitting procedure was carried out, the results can be saved to an Excel or NifTi file. Additionally, a heatmap
of the fitted pixels or ROIs can be created and saved as well.

> "When life gives you pineapples, make _tropical_ lemonade!"
> _- J. Jasse, Co-Founder and CTO of Pyneapple Global Ltd._

## Naming conventions

<center>

| variable   | assignment                                  |
|------------|---------------------------------------------|
| `d`        | diffusion coefficients                      |
| `f`        | volume fractions                            |
| `b_values` | b-values                                    |
| `S0`       | Signal at b = 0                             |
| `d_range`  | diffusion fitting range                     |
| `bins`     | log spaced values inside `d_range`          |
| `n_bins`   | number of `bins`                            |
| `x0`       | starting values for NLLS                    |
| `spectrum` | spectrum                                    |
| `img`      | MRI image                                   |
| `seg`      | segmentation/ROI of `img`                   |
| `model`    | one of the following: NLLS, IDEL, NNLS, ... |
| `n_pools`  | number of cpu kernels for multi-processing  |

</center>

---
v0.7.1
