from pathlib import Path
import time
import numpy as np
from multiprocessing import freeze_support

from src.utils import Nii, NiiSeg
from src.fit.fit import FitData

if __name__ == "__main__":
    start_time = time.time()
    freeze_support()
    # Test Set
    # img = Nii(Path(r"data/test_img_176_176.nii"))
    # seg = NiiSeg(Path(r"data/test_mask_simple_huge.nii.gz"))
    # json = Path(
    #     r"resources/fitting/default_params_ideal_test.json",
    # )
    # Prostate Set
    img = Nii(Path(r"data/01_prostate_img.nii"))
    seg = NiiSeg(Path(r"data/01_prostate_mask.nii.gz"))
    json_ideal = Path(
        r"resources/fitting/params_prostate_ideal.json",
    )
    ivim_json = Path(r"resources/fitting/params_prostate_ivim.json")

    img.zero_padding()
    # img.save("01_prostate_img_164x.nii.gz")
    seg.zero_padding()
    seg.array = np.fliplr(seg.array)
    # seg.save("01_prostate_seg_164x.nii.gz")

    # IDEAL
    data = FitData("IDEAL", json_ideal, img, seg)
    data.fit_params.n_pools = 12
    data.fit_ideal(multi_threading=True, debug=True)
    data.fit_results.save_results_to_nii(
        "test_ideal.nii", data.img.array.shape, dtype=float
    )
    stop_time = time.time() - start_time
    print(f"{round(stop_time, 2)}s")

    # IVIM
    data_ivim = FitData("IVIM", ivim_json, img, seg)
    data_ivim.fit_results.save_results_to_nii(
        "test_ivim.nii", data_ivim.img.array.shape, dtype=float
    )
    stop_time = time.time() - stop_time
    print(f"{round(stop_time, 2)}s")
    print("Done")
