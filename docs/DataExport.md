# Exporting Fit Results

There are several options for exporting fit results.
- hdf5
- nifti
- excel

The recommended format is **hdf5**. The data is stored in the same way it is handled internally. With a dictionary, of D and f values holding pixel ($(x,y,z)$) or segmentation ($n$) data. Numpy arrays are sparsed using the [*sparse*](https://github.com/pydata/sparse/) package. 
The structure of the dictionary is as follows:

- D: diffusion coefficients as arrays ($D_1 ... D_n$).
- f: fractions for each coefficients as arrays ($f_1 ... f_n$).
- S0: absolute signal intensity.
- raw: raw decay signal which is used to calculate the fit results.
- curve: decay curve calculated from the fit results.
- spectrum: diffusion spectrum for coefficients and fractions

Export is performed using the following methods:
``` python
results.save_to_hdf5('filename.h5')
# or for array instead of dictionary
results.save_to_hdf5_as_array('filename.npy', img:RadImgArray = img)
```
