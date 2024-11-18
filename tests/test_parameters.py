import pytest
from pathlib import Path
import numpy as np
from pyneapple.parameters.parameters import BaseParams


class TestParameters:
    def test_load_b_values(self, root):
        parameters = BaseParams()
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

    def test_get_pixel_args(self, img, seg):
        parameters = BaseParams()
        args = parameters.get_pixel_args(img, seg)
        assert len(list(args)) == len(np.where(seg != 0)[0])

    @pytest.mark.parametrize("seg_number", [1, 2])
    def test_get_seg_args_seg_number(self, img, seg, seg_number):
        parameters = BaseParams()
        args = parameters.get_seg_args(img, seg, seg_number)
        assert len(list(args)) == 1
