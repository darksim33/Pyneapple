## Documentation

# Define image and segmentation file
img = ut.Nii(Path(r"data/01_img.nii"))
seg = ut.Nii_seg(Path(r"data/01_img_seg_test.nii"))

# Define fitting object by specifying image, segmentation and model to be fitted
NNLS_fit = FitData("NNLS", img, seg)

# Fit fitting object pixelwise
NNLS_fit.fitting_pixelwise(debug=True)


# ------------------------------------------------------------------------------
## Workflow of Pyneapple:
# 1. Create fitting object NNLS_fit by naming the model and passing image and segmentation
NNLS_fit = FitData("NNLS", img, seg)

# 2. FitData initialises a fitting model with said model properties
class FitData:
    if model == "NNLS":
        self.fit_params = NNLSParams(FitModel.NNLS)
# and further basic fitting parameters (?) (model name, b_values, max_iter, fitting specifications, number of CPUs), a placeholder for future results and an option for different fitting mechanisms (?) (atm only pixelwise) and 

    # 2a. By creating a model using the Model class
    class Model:
        def NNLS(idx: int, signal: np.ndarray, basis: np.ndarray, max_iter: int = 200):
        fit, _ = nnls(basis, signal, maxiter=max_iter)
        return idx, fit
    # returning the fitted NNLS signal (and index)

    # 2b. Passed to the NNLSParams class:
    class NNLSParams: 
        ...
    # adding model specific parameters (number of bins, maximum iterations, diffusion range) and allowing for output of the b-values/basis (?), fitted signal function and evaluated fitting result (?)

# 3. Fitting is the done pixelwise
NNLS_fit.fitting_pixelwise()

    # 3a. Calling the global fit function
    def fit(fitfunc, pixel_args, n_pools, debug: bool | None = False):
    
    # 3b. And saving the results into the nested Results class:
    class FitData:
        class Results:
            ...
    # containing lists of lists holding evaluated parameters like d- and f-values and if applicable S0 and T1