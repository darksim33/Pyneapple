# from PineappleUI import startAppUI
from pathlib import Path
import time
from multiprocessing import freeze_support

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches  #
# from PIL import Image
from functools import partial

import numpy as np
from multiprocessing import freeze_support

from src.utils import Nii, NiiSeg
from src.fit.fit import FitData  # , imantics
from src.fit.parameters import IDEALParams
from src.fit.ideal import fit_ideal_new

# from plotting import Plot

if __name__ == "__main__":
    start_time = time.time()
    freeze_support()
    img = Nii(Path(r"data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"data/test_mask_simple_huge.nii.gz"))
    json = Path(
        r"resources/fitting/default_params_ideal_test.json",
    )
    ideal_params = IDEALParams(json)
    result = fit_ideal_new(img, seg, ideal_params, debug=False, multithreading=False)
    scaling = np.array([10000, 10000, 10000, 100, 100, 1])
    out_nii = Nii().from_array(result * scaling)
    out_nii.save("test.nii")
    print(f"{round(time.time() - start_time, 2)}s")
    print("Done")
