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
can define them in a [json or toml file](Parameters.md) and load the file.
There are two ways to handle the images, parameters and the actual fitting afterward. For basic usage you can use the 
```FitData``` class, which handles the fitting process and stores the results in a nested ```Results``` class. For more 
control over the fitting process you can use the underlying parameter classes directly and call the fitting and
evaluation functions yourself.

## Using FitData Class

```python
data = FitData(img=img, seg=seg, params_file="fitting_parameters_NNLS.json")
```
The *FitData* class will detect the selected fitting model and load the fitting parameters from the provided parameter file. 
Three different fitting options are available. The standard pixel-wise fitting will calculate the diffusion parameters
for every pixel in a ROI/segmentation. The segmentation-wise fitting will calculate the diffusion parameters for mean
signals of every ROI/segmentation. The IVIM segmented fitting will calculate the diffusion parameters for every pixel in
a two-step process. First the ADC parameters are fitted and then the higher order diffusion parameters are fitted using
the IVIM approach. 

```python
data.fit_pixel_wise(fit_type="multi")  # single, multi, gpu are supported 
data.fit_segmentation_wise()
data.fit_ivim_segmented(fit_type="multi") # single, multi, gpu are supported 
```
It is carried out by the ```fit``` module, which stores the results in the nested ```Results``` class. This object then
contains all evaluated diffusion parameters such as d- and f-values and results for S0 and T1, if applicable.

## Fitting directly

If you want to fine tune the processing you can use the underlying parameter classes directly and perform the fitting
and evaluation using the basic functions directly. This will return a list of tuples holding the pixel indexes and the
corresponding fit results. This way only the raw results are returned which can be further processed or visualized using
the ```Results``` class in the same way the ```FitData``` does.
 
```python
from pyneapple.parameters import NNLSParams
from pyneapple.fitting.fit import fit_pixel_wise

params = NNLSParams(params_file="fitting_parameters_NNLS.json")
result = fit_pixel_wise(img, seg, params, fit_type="multi") # single, multi, gpu are supported 
```
For the "direct fitting" the same options are available as for the ```FitData``` class with pixel-wise, 
segmentation-wise and IVIM segmented fitting. Both approaches are capable of using multi-threading and GPU acceleration
on Nvidia Cards. 
___
