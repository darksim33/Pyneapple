import pytest
from pathlib import Path
import numpy as np
from pyneapple import Parameters


class TestParameters:
    def test_load_b_values(self, root):
        parameters = Parameters()
        file = root / r"tests/.data/test_bvalues.bval"
        assert file.is_file()
        parameters.load_b_values(file)
        b_values = np.array(
            [
                0,
                50,
                100,
                150,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                1000,
                1200,
                1400,
                1600,
                1800,
            ]
        )
        assert b_values.all() == parameters.b_values.all()

    def test_get_bins(self):
        parameters = Parameters()
        bins = parameters.get_bins()
        assert bins.shape == (250,)
        assert bins.max() == 1
        assert bins.min() == 0.0001

    def test_get_pixel_args(self, img, seg):
        parameters = Parameters()
        args = parameters.get_pixel_args(img.array, seg.array)
        assert len(list(args)) == len(seg.seg_indices)

    @pytest.mark.parametrize("seg_number", [1, 2])
    def test_get_seg_args_seg_number(self, img, seg, seg_number):
        parameters = Parameters()
        args = parameters.get_seg_args(img.array, seg, seg_number)
        assert len(list(args)) == 1
