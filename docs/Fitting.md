# Using Pyneapple for DWI fitting

There are different ways of using Pyneapple for fitting diffusion-weighted images but all start by creating a script and
load the image and segmentation data using the ```RadImgArray``` and ```SegImgArray``` classes:

```python
from pathlib import Path
from radimgarray import RadImgArray, SegImgArray

img = RadImgArray(Path(r"image.nii"))
seg = SegImgArray(Path(r"segmentation.nii.gz"))
```
A common way to import these image files is to use the NifTi format, but dicom is supported as well.
Afterward you can either initialize a Parameter class and set the necessary fitting parameters inside your script, or you
can define them in a [json file](#The-Parameter-Json) and load the file to a parameter class or directly to the fitting data class.

Case 1:
```python
data = FitData(img=img, seg=seg, params_json="fitting_parameters_NNLS.json")
```
Case 2:
```python
params = NNLSParams(params_json="fitting_parameters_NNLS.json")
```

```FitData``` then initialises a [fitting model](#the-model-class) with loaded model properties, other (partially model
specific) fitting parameters such as b-values, maximum iterations, number of CPUs and [many more](#the-json-not-derulo)
and a placeholder for future results. If no fitting parameters are provided, the ```FitData``` class will initialise
default
parameters dependent on the chosen fitting model.

Fitting can either be done pixel-wise or for whole segmentation images:

```python
data.fit_pixel_wise()
data.fit_segmentation_wise(multi_threading=True)

# Optional applying of the AUC constraint:
d_AUC, f_AUC = data.params.apply_AUC_to_results(data.results)
```

It is carried out by the ```fit``` module, which stores the results in the nested ```Results``` class. This object then
contains all evaluated diffusion parameters such as d- and f-values and results for S0 and T1, if applicable.
Optionally,
a boolean can be passed to enable the multi-threading feature of Pyneapple (set ```True``` by default). After fitting,
an
AUC constraint can be applied to the results, for the NNLS_AUC fitting approach or for general AUC smoothing of the
acquired data.

Following the fitting procedure, the results can be saved to an Excel or NifTi file. Additionally, a heatmap
of the fitted pixels or ROIs can also be generated and saved.

___
