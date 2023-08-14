from multiprocessing import freeze_support
from pathlib import Path
from functools import partial
import numpy as np

from src.fit.ideal import IdealFitting as IDEAL
from src.fit.model import Model
from src.utils import Nii, NiiSeg


def test_ideal_triexp_multithreading():
    freeze_support()
    img = Nii(Path(r"../data/kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    ideal_params = IDEAL.IDEALParams()
    ideal_params.b_values = np.array(
        [
            [
                0,
                5,
                10,
                20,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
                250,
                300,
                400,
                525,
                750,
            ]
        ]
    )
    # ideal_params.model = partial(Model.multi_exp, n_components=3)
    ideal_params.model = Model.mono
    IDEAL.fit_ideal(img, ideal_params, seg)

    assert True
