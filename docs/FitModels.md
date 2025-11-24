# Fit Models 

All supported fit models and the corresponding settings are listed here.

## Multi-Exponential IVIM

### Basic Model 

Mono-Exponential:

$$ S(b) = \exp (-b_{values} \cdot D_1) \cdot S_0 $$

*The mono-exponential model uses $S_0$ instead of $f_1$*.

Bi-Exponential:

$$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2)$$

Tri-Exponential: 

$$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2) + f_3 \exp(-b_{values} \cdot D_3)$$

### Fit $S_0$ Models

> To use the model containg $S_0$ set *fit_s0* to *true* and add boundaries accordingly [*Example File*](../examples/parameters/params_biexp_s0.toml)
.

Bi-Exponential:

$$ S(b) = (f_1 \exp (-b_{values} \cdot D_1) + (1 - f_1) \exp(-b_{values} \cdot D_2)) * S_0$$

Tri-Exponential: 

$$ S(b) = (f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2) + (1 - f_1 - f_2) \exp(-b_{values} \cdot D_3)) * S_0$$

### Reduced Models

> To use the reduced fitting model set *fit_reduced* to *true* and remove the related fraction or $S_0$ boundaries. [*Example File*](../examples/parameters/params_triexp_reduced.toml)

Mono-Exponential:

$$ S(b) = \exp (-b_{values} \cdot D_1)$$

Bi-Exponential:

$$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + (1 - f_1) \exp(-b_{values} \cdot D_2)$$

Tri-Exponential: 

$$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2) + (1 - f_1 - f_2) \exp(-b_{values} \cdot D_3)$$

### With $T_1$ Fitting

All models get an extra $T_1$ Fitting Term at the end of the equation.
>  For fitting $T_1$ aswell set *fit_t1* to true and add related boundaries. [*Example File*](../examples/parameters/params_biexp_t1.toml)

Mono-Exponential: 

$$ S(b) = \exp (-b_{values} \cdot D_1) \cdot S_0 \cdot \exp (\frac{-T_1}{t_{mix}})$$

## NNLS Fitting

The NNLS algorithm performs a fitting without knowing the exact number of components present in the data.

Data points $y_i$ are modeled using fractions $s_i$ and the matrix containing the exponentials $A_{ij}$.

$$ y_i = \sum^{M}_{j=1} A_{ij} s_{j} \qquad i = 1,2,\ldots, N$$

The algorithm tries to minimize the error for model and data.

$$ \chi^{2} = min \left[ \sum^{N}_{i=1} \left\vert \sum^{M}_{j=1} A_{ij}s_j - y_i \right\vert^{2} \right] $$

It's also prossible to use regularization term to get a more physilogical representation of the signal. For the second order this results in:

$$ \chi^{2} = min \left[ \sum^{N}_{i=1} \left\vert \sum^{M}_{j=1} A_{ij}s_j - y_i \right\vert^{2} + \mu \sum^{M}_{j=1} \vert s_{j+2} - 2 s_j+1 + s_j \vert^{2}  \right] $$

For more information see: [Whittall (1989)](https://doi.org/10.1016/0022-2364(89)90011-5), [Lawson (1995)](https://doi.org/10.1137/1.9781611971217), [Stabinska (2021)](https://doi.org/10.1002/MRM.28631), [Periquito (2021)](https://doi.org/10.21037/qims-20-1360), [Bjarnason (2010)](https://doi.org/10.1016/j.jmr.2010.07.008).