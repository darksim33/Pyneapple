# PyneappleUI

> "Pineapple is one of my favorite fruits. Just not on pizza."  
> _- T. Thiel, Co-Founder and CEO of Pyneapple Global Ltd._

## Description
The PyneappleUI is an advanced tool to analyse multi-exponential signal data in MR DWI images. It is able to apply a variety of different fitting algorithms to the measured diffusion data and compare multi-exponential fitting methods. It can determine the total number of components contributing to the corresponding multi-exponential signal and analyses the results by calculating the corresponding diffusion parameters. Fitting can be customised. It can be done pixel by pixel or for whole segmentations.

> "A juicy tropical delight, an exquisite golden nectar of the tropics. Just as this code is exquisite and golden nectar of pure genius."  
> _- Steve Jobs, Former chairman and CEO of another, similarly successful fruit company_

## Workflow of Pyneapple

After defining an image and segmentation file
```python
img = utils.Nii(Path(r"img.nii"))
seg = utils.Nii_seg(Path(r"seg.nii"))
```
a fitting object is created by specifying the desired model, e.g. the NNLS model, and passing the image and segmentation:
```python
data_fit = FitData("NNLS", img, seg)
```
FitData initialises a fitting model with said model properties, further basic fitting parameters (model name, b_values, max_iter, fitting specifications, number of CPUs), a placeholder for future results and an option for different fitting types (pixel by pixel or whole segmentations).

By creating a model using the Model class 
 ```python
class Model:
    def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int = 200):
    fit, _ = nnls(basis, signal, maxiter=max_iter)
    return idx, fit
```
it returns the model-specific fit of the signal and passes it to the corresponding parameter class (in this case ```NNLSParams```)
```python
class FitData:
    if model == "NNLS":
        self.fit_params = NNLSParams(FitModel.NNLS)
```
which adds default model-specific parameters (e.g. number of bins, maximum iterations, diffusion range) and allows manipulation as well as output of the different fitting characteristics and parameters.

Fitting can either be done pixelwise or for whole segmentation images. Fitting is carried out by the ```fit``` function, saving the results into the nested Results class ```class Results```. This object then contains all evaluated diffusion parameters like d- and f-values and numbers for S0 and T1, if applicable.

## Naming conventions
<center>

| variable   | assignment                         |
| ---------- | ---------------------------------- |
| `d`        | diffusion coefficients             |
| `f`        | volume fractions                   |
| `b_values` | b-values                           |
| `S0`       | Signal for b = 0                   |
| `d_range`  | diffusion fitting range            |
| `bins`     | log spaced values inside `d_range` |
| `n_bins`   | number of `bins`                   |
| `x0`       | starting values                    |
| `spectrum` | spectrum                           |
| `img`      | MRI image                          |
| `seg`      | segmentation/ROI of `img`          |
| `model`    | one of the following: NLLS, NNLS, ...|
| `n_pools`  | number of cpu kernels for multi-processing |

</center>
---
v0.41  