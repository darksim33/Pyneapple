from multiprocessing import freeze_support
from pathlib import Path
from functools import partial

from src.fit.ideal import ideal_fitting as IDEAL
from src.fit.model import Model
from src.utils import Nii, Nii_seg


def test_ideal_triexp_multithreading():
    freeze_support()
    img = Nii(Path(r"data/01_img.nii"))
    seg = Nii_seg(Path(r"data/01_prostate.nii.gz"))
    ideal_params = IDEAL.IDEALParams()
    ideal_params.model = partial(Model.multi_exp(n_components=3))
    IDEAL.fit_ideal(img, ideal_params, seg)

    assert True
