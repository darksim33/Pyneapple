# from PineappleUI import startAppUI
from pathlib import Path

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches  #
# from PIL import Image
from functools import partial

import numpy as np
from multiprocessing import freeze_support

import src.utils as ut
from src.fit.fit import FitData  # , imantics
from src.fit.ideal import ideal_fitting as IDEAL
from src.fit.model import Model

# from plotting import Plot

if __name__ == "__main__":
    freeze_support()
    img = ut.Nii(Path(r"data/01_img.nii"))
    seg = ut.Nii_seg(Path(r"data/01_prostate.nii.gz"))
    ideal_params = IDEAL.IDEALParams()
    ideal_params.model = partial(Model.multi_exp, n_components=3)
    IDEAL.fit_ideal(img, ideal_params, seg)

    # fit_data = FitData("NNLSreg", img, seg)
    # fit_data.fit_params.max_iter = 10000
    # fit_data.fit_params.reg_order = 3
    # # fit_data.fit_pixel_wise()
    # results = fit_data.fit_segmentation_wise()
    print("Done")
