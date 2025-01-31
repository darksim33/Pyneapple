# The Parameter Json

By writing all relevant fitting parameters into a json file, correct fitting of the image data and storage of your
fitting parameters is ensured. Due to strong dependencies on initial fitting parameters in some of the implemented
approaches, it is strongly recommended to use a specified json file with an adapted parameter set for each model
(and image region). The json file can contain the following basic fitting parameters:

| name             | description                                | value                                                              |
| :--------------- | :----------------------------------------- | :----------------------------------------------------------------- |
| ```Class```      | corresponding parameter class              | "IVIMParams",  "IVIMSegmentedParams", "NNLSParams", "NNLSCVParams" |
| ```model```      | fitting model used                         | "monoexp", "biexp", "biexp_red", ... "nnls"                        |
| ```fit_type```   | fitting approach used                      | "single", "multi", "gpu"                                           |
| ```b-values```   | used for imaging                           | list of ints                                                       |
| ```max_iter```   | maximum iterations                         | int                                                                |
| ```n_pools```    | number of pools (CPUs) for multi-threading | int                                                                |
| ```boundaries``` | model specific fitting boundaries          | dict                                                               |

## Model Specific parameters

Every model has some specific parameters including the boundaries. 

### IVIM

#### Boundaries
The boundaries value ist a dictionary holding sub dictionaries for each component. For IVIM there are generally two options. Ether the *S0* signal is fitted within the fractions of the components or the
decays are scale using *S/S0* (*red* fitting) and relativ fractions are calculated. For the later case the "fastest" component is always calculated from the others and thereby dropped from the calculation. 

Example for reduced bi-exponential:
``` json
"boundaries": {
    "D": {
      "slow": [...],
      "fast": [...]
    },
    "f": {
      "slow": [...]
    }
  }
```

#### Segmented Fitting
For the segmented fitting an additional *options* section is added containing a dictionary. 
Some exemplary parameter files can be found [here](./tests/.data/fitting).

| name               | description                                                | type       |
| ------------------ | ---------------------------------------------------------- | ---------- |
| "fixed_component"  | Name of component to fix                                   | str        |
| "fixed_t1"         | whether T1 should be calculated in the first step          | bool       |
| "reduced_b_values" | can be *false* or a list of *b_values* for initial fitting | bool, list |


### NNLS

Since the NNLS method takes a different approach in calculating the underlying diffusion parameters a different set of boundaries and additional parameters is needed.

Boundaries:

| name      | description                                  | type               |     |
| --------- | -------------------------------------------- | ------------------ | --- |
| "d_range" | diffusion value range for NNLS fitting       | list(float, float) |     |
| "n_bins"  | number of exponential terms used for fitting | int                |     |

Additionally, regularization order (*reg_order*) and factor (*mu*) are needed. The first can ether be 0 for no regularization, 1-3 for different orders or "CV" for cross validation. 

___
