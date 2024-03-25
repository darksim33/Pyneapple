from pathlib import Path
from pyneapple.utils import Nii, NiiSeg


def test_nii():
    img = Nii(Path(r"../data/test_img.nii"))
    if img.path:
        assert True
    else:
        assert False


def test_nii_zero_padding():
    img = Nii(Path(r"../data/test_img_176_128.nii"), do_zero_padding=True)
    if img.path:
        assert True
    else:
        assert False


def test_nii_seg():
    seg = NiiSeg(Path(r"../data/test_mask_128.nii.gz"))
    if seg.path:
        assert True
    else:
        assert False
