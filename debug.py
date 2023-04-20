# from PineappleUI import startAppUI
from pathlib import Path
import utils as ut
import fitting
from multiprocessing import freeze_support

# p = Path(r"C:\Users\thitho01\Documents\Python\Projects\NNLSDynAPP\data\pat01_img.nii")
# startAppUI(p)

if __name__ == "__main__":
    freeze_support()

nii = Path(r"data/test9AV.nii")
img = ut.nifti_img(nii)
nii = Path(r"data/test9AVmask.nii")
mask = ut.nifti_img(nii)
fit_params = fitting.fitParameters()
fit_params.fitModel = fitting.fitModels.NNLSreg
fit_params.boundries.lb = 1 * 1e-4
fit_params.boundries.ub = 2 * 1e-1
fit_params.boundries.nbins = 250
fit_params.nPools = 4
result = fitting.setupFitting(img, mask, fit_params, True)
out = ut.nifti_img().fromArray(result)

print("test")
