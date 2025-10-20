# Fit Examples

## IVIM Fitting

To Run the classic bi-exponential IVIM fitting on a single data set you can use the provided example script with the [paramter file](../examples/parameters/params_biexp.json). For the toml file [see](../examples/parameters/params_biexp.toml). 

With Model:
$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2)$

[**IVIM Bi-Exponential Script**](../examples/ivim_fitdata_script.py)


## NNLS Fitting

For running a *NNLS* fitting without regularisation using the [parameter file](../examples/parameters/params_nnls.toml) use the following script.

[**NNLS Script**](../examples/nnls_fitdata_script.py)