from pathlib import Path
from multiprocessing import freeze_support

from src.utils import Nii, Nii_seg
from src.fit import fit


def test_nnls_multithreading():
    freeze_support()
    img = Nii(Path(r"data/01_img.nii"))
    seg = Nii_seg(Path(r"data/01_prostate.nii.gz"))

    fit_data = fit.FitData("NNLS", img, seg)
    fit_data.fit_params.max_iter = 10000
    fit_data.fit_params.reg_order = 3
    fit_data.fit_pixel_wise()
    # results = fit_data.fitting_segmentation_wise(seg_number=1)
    assert True
