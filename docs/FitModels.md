# Fit Models 

All supported fit models and the corresponding settings are listed here.

## Multi-Exponential IVIM

### Basic Model 

Mono-Exponential:

$ S(b) = \exp (-b_{values} \cdot D_1) \cdot S_0 $

*The mono-exponential model uses $S_0$ instead of $f_1$*.

Bi-Exponential:

$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2)$

Tri-Exponential: 

$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2) + f_3 \exp(-b_{values} \cdot D_3)$

### Fit $S_0$ Models

> To use the model containg $S_0$ set *fit_s0* to *true* and add boundaries accordingly [*Example File*](../examples/parameters/params_biexp_s0.toml)
.

Bi-Exponential:

$ S(b) = (f_1 \exp (-b_{values} \cdot D_1) + (1 - f_1) \exp(-b_{values} \cdot D_2)) * S_0$

Tri-Exponential: 

$ S(b) = (f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2) + (1 - f_1 - f_2) \exp(-b_{values} \cdot D_3)) * S_0$

### Reduced Models

> To use the reduced fitting model set *fit_reduced* to *true* and remove the related fraction or $S_0$ boundaries. [*Example File*](../examples/parameters/params_triexp_reduced.toml)

Mono-Exponential:

$ S(b) = \exp (-b_{values} \cdot D_1)$

Bi-Exponential:

$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + (1 - f_1) \exp(-b_{values} \cdot D_2)$

Tri-Exponential: 

$ S(b) = f_1 \exp (-b_{values} \cdot D_1) + f_2 \exp(-b_{values} \cdot D_2) + (1 - f_1 - f_2) \exp(-b_{values} \cdot D_3)$

### With $T_1$ Fitting

All models get an extra $T_1$ Fitting Term at the end of the equation.
>  For fitting $T_1$ aswell set *fit_t1* to true and add related boundaries. [*Example File*](../examples/parameters/params_biexp_t1.toml)

Mono-Exponential: 

$ S(b) = \exp (-b_{values} \cdot D_1) \cdot S_0 \cdot \exp (\frac{-T_1}{t_{mix}})$ 