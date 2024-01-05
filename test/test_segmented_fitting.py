import pytest
from pathlib import Path
from multiprocessing import freeze_support
import numpy as np

from src.fit.parameters import IVIMParams
from src.utils import Nii, NiiSeg
from src.fit import fit


# TODO: checking for the correct calculation of the mean segmentation signal instead of fitting?!
@pytest.fixture
def nnls_fit_data():
    img = Nii(Path(r"../data/01_img.nii"))
    seg = NiiSeg(Path(r"../data/01_prostate.nii.gz"))

    fit_data = fit.FitData(
        "NNLS", Path("resources/fitting/default_params_NNLS.json"), img, seg
    )
    fit_data.fit_params.max_iter = 10000

    return fit_data


@pytest.fixture
def tri_exp():
    img = Nii(Path(r"../data/kid_img.nii"))
    seg = NiiSeg(Path(r"../data/kid_mask.nii"))
    fitData = fit.FitData(
        "TriExp", Path("resources/fitting/default_params_IVIM.json"), img, seg
    )
    fitData.fit_params = IVIMParams()

    fitData.fit_params.boundaries.x0 = np.array(
        [
            0.1,  # D_fast
            0.005,  # D_inter
            0.0015,  # D_slow
            0.1,  # f_fast
            0.2,  # f_inter
            210,  # S_0
        ]
    )
    fitData.fit_params.boundaries.lb = np.array(
        [
            0.01,  # D_fast
            0.003,  # D_intermediate
            0.0011,  # D_slow
            0.01,  # f_fast
            0.1,  # f_inter
            10,  # S_0
        ]
    )
    fitData.fit_params.boundaries.ub = np.array(
        [
            0.5,  # D_fast
            0.01,  # D_inter
            0.003,  # D_slow
            0.7,  # f_fast
            0.7,  # f_inter
            1000,  # S_0
        ]
    )
    return fitData


def test_nnls_segmented_reg_0(fit_data):
    fit_data.fit_params.reg_order = 0
    fit_data.fit_segmentation_wise()

    nii_dyn = Nii().from_array(fit_data.fit_results.spectrum)
    nii_dyn.save(r"nnls_seg_seq_reg0.nii")
    assert True


def test_tri_exp_segmented(tri_exp):
    tri_exp.fit_segmentation_wise()
    assert True
