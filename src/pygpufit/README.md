# GPUfit

Enables GPU based fitting on Cuda capable GPUs.

## Update GPUfit

For adding or altering fitting models use the [GPUfit](https://github.com/darksim33/GPUfit) repository.
Compile the GPUfit package to get the necessary .dll or .so files and copy them to the src folder.
Each new model or algorithm needs to be added to "gpufit.py".

## Models

### Mono-Exponential

Basic:
$y = a*exp(-b*x)$ with (a: f1, b: D1)

Reduced:
$S/S0 = exp(-a*x)$ with (a: D1)

T1:

### Bi-Exponential

Basic:
$y = a*exp(-b*x) + c*exp(-d*x)$ with (a: f1, b: D1, c: f2, d: D2)

Reduced:
$S/S0 = a*exp(-b*x) + (1-a)*exp(-c*x)$ with (a: f1, b: D1, c: D2)

T1:

### Tri-Exponential

Basic:
$y = a*exp(-b*x) + c*exp(-d*x) + e*exp(-f*x)$ with (a: f1, b: D1, c: f2, d: D2, e: f3, f: D3)

Reduced:
$S/S0 = a*exp(-b*x) + c*exp(-d*x) + (1-a-c)*exp(-e*x)$ with (a: f1, b: D1, c: f2, d: D2, e: D3)

T1:
