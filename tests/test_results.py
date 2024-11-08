import pandas as pd
import numpy as np
import pytest

from pyneapple import Results
from pyneapple import Parameters, IVIMParams


def test_custom_dict_validate_key():
    results = Results(Parameters())
    results.spectrum.set_segmentation_wise({(1, 1, 1): 1})
    results.spectrum[np.int32(1)] = 1.1
    for key in results.spectrum:
        assert isinstance(key, int)


def test_results_update():
    f = {1: [1.1, 1.2, 1.3]}
    d = {1: [1.1, 1.2, 1.3]}
    results = Results(Parameters())
    results.update_results({"d": d, "f": f})
    assert results.d == d
    assert results.f == f


def test_results_set_seg_wise():
    pixel2seg = {(1, 1, 1): 1, (1, 1, 1): 1}
    f = {1: [1.1, 1.2, 1.3]}
    d = {1: [1.1, 1.2, 1.3]}
    results = Results(Parameters())
    results.f.update(f)
    results.d.update(d)
    results.set_segmentation_wise(pixel2seg)
    assert results.f[1, 1, 1] == f[1]
    assert results.d[1, 1, 1] == d[1]
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


def compare_lists_of_floats(list_1: list, list_2: list):
    """Compares lists of floats"""
    list_1 = [round(element, 10) for element in list_1]
    list_2 = [round(element, 10) for element in list_2]
    assert list_1 == list_2


def test_save_spectrum_to_excel(array_result, out_excel):
    result = Results(Parameters())
    save_spectrum_to_excel(array_result, out_excel, result)


def save_spectrum_to_excel(array_result, out_excel, result):
    for idx in np.ndindex(array_result.shape[:-2]):
        spectrum = array_result[idx]
        result.spectrum.update({idx: spectrum})

    # if out_excel.is_file():
    #     out_excel.unlink()
    bins = np.linspace(0, 10, 11)
    result.save_spectrum_to_excel(out_excel, bins)
    df = pd.read_excel(out_excel, index_col=0)
    columns = df.columns.tolist()
    assert columns == ["pixel"] + bins.tolist()
    for idx, key in enumerate(result.spectrum.keys()):
        spectrum = np.array(df.iloc[idx, 1:])
        compare_lists_of_floats(
            spectrum.tolist(), np.squeeze(result.spectrum[key]).tolist()
        )


def test_save_fit_curve_to_excel(array_result, out_excel):
    result = Results(Parameters())
    save_curve_to_excel(array_result, out_excel, result)


def save_curve_to_excel(array_result, out_excel, result):
    for idx in np.ndindex(array_result.shape[:-2]):
        curve = array_result[idx]
        result.curve.update({idx: curve})

    b_values = np.linspace(0, 10, 11).tolist()
    result.save_fit_curve_to_excel(out_excel, b_values)
    df = pd.read_excel(out_excel, index_col=0)
    columns = df.columns.tolist()
    assert columns == ["pixel"] + b_values
    for idx, key in enumerate(result.curve.keys()):
        curve = np.array(df.iloc[idx, 1:])
        compare_lists_of_floats(curve.tolist(), np.squeeze(result.curve[key].tolist()))
