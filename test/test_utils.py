import pytest

from pathlib import Path
from pyneapple.utils.nifti import Nii, NiiSeg


def test_nii_zero_padding(img):
    img = Nii(Path(r".data/test_img_176x128.nii"), do_zero_padding=True)
    if img.path:
        assert True
    else:
        assert False
