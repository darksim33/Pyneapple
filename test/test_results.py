import pytest
import pandas as pd

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


def test_results_update():
    f = {1: [1.1, 1.2, 1.3]}
    d = {1: [1.1, 1.2, 1.3]}
    results = Results()
    results.update_results({"d": d, "f": f})
    assert results.d == d
    assert results.f == f


def test_results_set_seg_wise():
    pixel2seg = {(1, 1, 1): 1, (1, 1, 1): 1}
    f = {1: [1.1, 1.2, 1.3]}
    d = {1: [1.1, 1.2, 1.3]}
    results = Results()
    results.f.update(f)
    results.d.update(d)
    results.set_segmentation_wise(pixel2seg)
    assert results.f[1, 1, 1] == f[1]
    assert results.d[1, 1, 1] == d[1]
    assert results.f.identifier == pixel2seg


def test_save_to_excel(nnls_fit_results_data, out_excel):
    if out_excel.is_file():
        out_excel.unlink()
    # basic
    nnls_fit_results_data.save_results_to_excel(
        out_excel, split_index=False, is_segmentation=False
    )
    assert out_excel.is_file()
    df = pd.read_excel(out_excel)
    assert df.columns.tolist() == 1
