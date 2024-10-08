import pandas as pd

from pathlib import Path
from nifti import Nii
from test_results import compare_lists


def test_nii(img):
    assert isinstance(img, Nii)


def test_nii_zero_padding(img):
    img = Nii(Path(r".data/test_img_176x128.nii"), do_zero_padding=True)
    if img.path:
        assert True
    else:
        assert False


def test_save_mean_signal_to_excel(img, seg, nnls_params, out_excel):
    if out_excel.is_file():
        out_excel.unlink()

    seg.save_mean_signals_to_excel(img, nnls_params.b_values, out_excel)
    df = pd.read_excel(out_excel)
    columns = df.columns.tolist()
    if len(columns) > 16:
        columns = columns[1:]
    b_values = nnls_params.b_values.squeeze().tolist()
    compare_lists(columns, b_values)
