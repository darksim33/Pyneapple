# from PineappleUI import startAppUI
from pathlib import Path
import utils as ut
from fit import *

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches  #
# from PIL import Image

# from plotting import Plot
# import numpy as np
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()

    # Define image and segmentation file
    img = ut.Nii(Path(r"data/01_img.nii"))
    seg = ut.Nii_seg(Path(r"data/01_img_seg_test.nii"))
    # dyn = ut.Nii(Path(r"data/01_img_AmplDyn.nii"))

    # Define fitting object by specifying image, segmentation and model to be fitted
    NNLS_fit = FitData("NNLS", img, seg)

    # Fit fitting object pixelwise
    NNLS_fit.fitting_pixelwise(debug=True)

    NNLSreg_fit = FitData("NNLSreg", img, seg)
    NNLSreg_fit.fitting_pixelwise(debug=True)

    print("Done")
