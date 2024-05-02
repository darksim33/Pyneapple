import pytest

from pyneapple.fit import Results


def test_custom_dict_get():
    results = Results()
    results.spectrum[(1, 1, 1)] = 1.1

    assert results.spectrum[(1, 1, 1)] == 1.1
    assert results.spectrum.get((1, 1, 1), 0) == 1.1


def test_custom_dict_get_seg():
    results = Results()
    results.spectrum.set_segmentation_wise({(1, 1, 1): 1})
    results.spectrum[1] = 1.1

    assert results.spectrum[(1, 1, 1)] == 1.1
    assert results.spectrum.get((1, 1, 1), 0) == 1.1
