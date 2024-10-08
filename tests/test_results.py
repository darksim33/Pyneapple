import pandas as pd
import numpy as np

from pyneapple import Results


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


def test_custom_dict_validate_key():
    results = Results()
    results.spectrum.set_segmentation_wise({(1, 1, 1): 1})
    results.spectrum[np.int32(1)] = 1.1
    for key in results.spectrum:
        assert isinstance(key, int)


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
    df = pd.read_excel(out_excel, index_col=0)
    assert df.columns.tolist() == [
        "pixel_index",
        "element",
        "D",
        "f",
        "compartment",
        "n_compartments",
    ]
    # Check if all items are writen to the df
    n_total_components = sum(len(array) for array in nnls_fit_results_data.d.values())
    assert n_total_components == df.shape[0]
    # Check split index
    out_excel.unlink()
    nnls_fit_results_data.save_results_to_excel(
        out_excel, split_index=True, is_segmentation=False
    )
    df = pd.read_excel(out_excel, index_col=0)
    assert out_excel.is_file()
    assert df.columns.tolist() == [
        "pixel_x",
        "pixel_y",
        "slice",
        "element",
        "D",
        "f",
        "compartment",
        "n_compartments",
    ]
    n_total_components = sum(len(array) for array in nnls_fit_results_data.d.values())
    assert n_total_components == df.shape[0]


def compare_lists(list_1: list, list_2: list):
    """Compares lists of floats"""
    list_1 = [round(element, 10) for element in list_1]
    list_2 = [round(element, 10) for element in list_2]
    assert list_1 == list_2


def test_save_spectrum_to_excel(nnls_fit_results_data, nnls_params, out_excel):
    if out_excel.is_file():
        out_excel.unlink()

    nnls_fit_results_data.save_spectrum_to_excel(nnls_params.get_bins(), out_excel)
    df = pd.read_excel(out_excel, index_col=0)
    columns = df.columns.tolist()
    bins = nnls_params.get_bins().tolist()
    compare_lists(columns, bins)

    spectrum_orig = nnls_fit_results_data.spectrum[
        list(nnls_fit_results_data.spectrum.keys())[0]
    ]
    spectrum_excel = np.array(df.iloc[0])
    assert spectrum_orig.all() == spectrum_excel.all()


def test_save_fit_curve_to_excel(nnls_fit_results_data, nnls_params, out_excel):
    if out_excel.is_file():
        out_excel.unlink()

    nnls_fit_results_data.save_fit_curve_to_excel(nnls_params.b_values, out_excel)
    df = pd.read_excel(out_excel, index_col=0)
    columns = df.columns.tolist()
    b_values = nnls_params.b_values.squeeze().tolist()
    compare_lists(columns, b_values)

    curve_orig = nnls_fit_results_data.curve[
        list(nnls_fit_results_data.curve.keys())[0]
    ]
    curve_excel = np.array(df.iloc[0])
    assert curve_orig.all() == curve_excel.all()
