# The Model class

By creating a model using the ```Model``` class

 ```python
class Model:
    class NNLS(object):
        @staticmethod
        def fit(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int | None):
            fit, _ = nnls(basis, signal, maxiter=max_iter)

            return idx, fit
 ```

it returns the model-specific fit of the signal and passes it to the corresponding parameter class (in this
case ```NNLSParams```) which adds default model-specific parameters (e.g. number of bins, maximum iterations,
diffusion range) and allows manipulation and output of the different fitting characteristics and parameters.

___