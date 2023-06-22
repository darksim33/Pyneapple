# from PineappleUI import startAppUI
from pathlib import Path
import utils as ut
from fitting import *  # , imantics

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches  #
# from PIL import Image

# from plotting import Plot
# import numpy as np
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    img = ut.Nii(Path(r"data/01_img.nii"))
    # img = ut.Nii(Path(r"data/pat16_img.nii.gz"))
    # seg = ut.Nii_seg(Path(r"data/pat16_seg_test.nii.gz"))
    seg = ut.Nii_seg(Path(r"data/01_mask_test.nii"))
    # dyn = ut.Nii(Path(r"data/01_img_AmplDyn.nii"))

    fit = FitData("NNLSreg", img, seg)
    fit.fitting_pixelwise(debug=True)

    fit = FitData("NNLS", img, seg)
    fit.fitting_pixelwise(debug=True)

    print("Done")
