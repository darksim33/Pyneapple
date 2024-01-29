from pathlib import Path
import time

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
    # img = Nii(Path(r"data/01_prostate_img.nii"))
    # seg = NiiSeg(Path(r"data/01_prostate_mask.nii.gz"))
    img = Nii(Path(r"data/01_img.nii"))
    seg = NiiSeg(Path(r"data/01_prostate.nii.gz"))
    json = Path(
        r"resources/fitting/params_prostate_ideal.json",
    )
    img.zero_padding()
    seg.zero_padding()
    data = FitData("IDEAL", json, img, seg)
    data.fit_params.n_pools = 12
    data.fit_ideal(multi_threading=True)
    data.fit_results.save_results_to_nii("test.nii", data.img.array.shape)
    print(f"{round(time.time() - start_time, 2)}s")
    print("Done")
