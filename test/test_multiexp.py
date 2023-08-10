from multiprocessing import freeze_support
from pathlib import Path
import numpy as np

from src.utils import Nii, Nii_seg
from src.fit import fit
from src.fit.model import Model
from src.fit.parameters import MultiTest


def test_triexp_model_sequential():
    freeze_support()
    img = Nii(Path(r"../data/kid_img.nii"))
    seg = Nii_seg(Path(r"../data/kid_mask.nii"))
    fit_data = fit.FitData("TriExp", img, seg)
    fit_data.fit_params = MultiTest()
    fit_data.fit_params.boundaries.x0 = np.array(
        [
            0.1,  # D_fast
            0.005,  # D_interm
            0.0015,  # D_slow
            0.1,  # f_fast
            0.2,  # f_interm
            210,  # S_0
        ]
    )
    fit_data.fit_params.boundaries.lb = np.array(
        [
            0.01,  # D_fast
            0.003,  # D_intermediate
            0.0011,  # D_slow
            0.01,  # f_fast
            0.1,  # f_interm
            10,  # S_0
        ]
    )
    fit_data.fit_params.boundaries.ub = np.array(
        [
            0.5,  # D_fast
            0.01,  # D_interm
            0.003,  # D_slow
            0.7,  # f_fast
            0.7,  # f_interm
            1000,  # S_0
        ]
    )
    fit_data.fit_pixel_wise(multi_threading=False)

    assert True


# def test_triexp_model_multithreading():
#     freeze_support()
#     img = Nii(Path(r"../data/kid_img.nii"))
#     seg = Nii_seg(Path(r"../data/kid_mask.nii"))
#     fit_data = fit.FitData("TriExp", img, seg)
#     fit_data.fit_params = MultiTest()
#     fit_data.fit_params.boundaries.x0 = np.array(
#         [
#             0.1,  # D_fast
#             0.005,  # D_interm
#             0.0015,  # D_slow
#             0.1,  # f_fast
#             0.2,  # f_interm
#             210,  # S_0
#         ]
#     )
#     fit_data.fit_params.boundaries.lb = np.array(
#         [
#             0.01,  # D_fast
#             0.003,  # D_intermediate
#             0.0011,  # D_slow
#             0.01,  # f_fast
#             0.1,  # f_interm
#             10,  # S_0
#         ]
#     )
#     fit_data.fit_params.boundaries.ub = np.array(
#         [
#             0.5,  # D_fast
#             0.01,  # D_interm
#             0.003,  # D_slow
#             0.7,  # f_fast
#             0.7,  # f_interm
#             1000,  # S_0
#         ]
#     )
#     fit_data.fit_pixel_wise(multi_threading=True)
