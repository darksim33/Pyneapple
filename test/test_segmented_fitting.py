import pytest
import numpy as np

from pathlib import Path

from src.pyneapple.fit.parameters import IVIMParams
from pyneapple.utils.nifti import Nii, NiiSeg
from pyneapple.fit import FitData


# def test_ideal_tri_segmented(ideal_tri_fit_data: FitData, capsys):
#     ideal_tri_fit_data.fit_segmentation_wise()
#     capsys.readouterr()
#     assert True
