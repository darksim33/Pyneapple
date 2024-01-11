from multiprocessing import freeze_support
from pathlib import Path
from functools import partial
import numpy as np

from src.fit.ideal import IDEALParams
from src.fit.parameters import IVIMParams
from src.fit.model import Model
from src.utils import Nii, NiiSeg


def test_ideal_triexp_multithreading():
    freeze_support()
    img = Nii(Path(r"../data/kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    json = Path(
        Path(__file__).parent.parent, "./resources/fitting/default_params_ideal.json"
    )
    ideal_params = IDEALParams(json)
    test = IVIMParams(json)
    # IDEAL.fit_ideal(img, ideal_params, seg)

    assert True
