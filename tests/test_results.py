import pandas as pd
import numpy as np
import pytest

from pyneapple.results.results import BaseResults
from pyneapple import IVIMParams
from pyneapple.parameters.parameters import BaseParams
from .test_toolbox import ResultTools as Tools


def test_custom_dict_validate_key():
    results = BaseResults(BaseParams())
    results.spectrum.set_segmentation_wise({(1, 1, 1): 1})
    results.spectrum[np.int32(1)] = 1.1
    for key in results.spectrum:
        assert isinstance(key, int)


def test_results_update():
    f = {1: [1.1, 1.2, 1.3]}
    d = {1: [1.1, 1.2, 1.3]}
    results = BaseResults(BaseParams())
    results.update_results({"D": d, "f": f})
    assert results.D == d
    assert results.f == f


def test_results_set_seg_wise():
    pixel2seg = {(1, 1, 1): 1, (1, 1, 1): 1}
    f = {1: [1.1, 1.2, 1.3]}
    d = {1: [1.1, 1.2, 1.3]}
    results = BaseResults(BaseParams())
    results.f.update(f)
    results.D.update(d)
    results.set_segmentation_wise(pixel2seg)
    assert results.f[1, 1, 1] == f[1]
    assert results.D[1, 1, 1] == d[1]
    assert results.f.identifier == pixel2seg


def test_save_to_excel(random_results, out_excel):
    if out_excel.is_file():
        out_excel.unlink()
    # basic
    random_results.save_to_excel(out_excel, split_index=False, is_segmentation=False)
    assert out_excel.is_file()
    df = pd.read_excel(out_excel, index_col=0)
    assert df.columns.tolist() == [
        "pixel",
        "parameter",
        "value",
    ]
    # Check split index
    out_excel.unlink()
    random_results.save_to_excel(out_excel, split_index=True, is_segmentation=False)
    df = pd.read_excel(out_excel, index_col=0)
    assert out_excel.is_file()
    assert df.columns.tolist() == [
        "x",
        "y",
        "z",
        "parameter",
        "value",
    ]


def test_save_spectrum_to_excel(array_result, out_excel):
    result = BaseResults(BaseParams())
    Tools.save_spectrum_to_excel(array_result, out_excel, result)


def test_save_fit_curve_to_excel(array_result, out_excel):
    result = BaseResults(BaseParams())
    Tools.save_curve_to_excel(array_result, out_excel, result)
