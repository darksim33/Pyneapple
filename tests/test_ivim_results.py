import pandas as pd
import numpy as np

from pyneapple import IVIMResults


def test_eval_results(ivim_bi_params, results_bi_exp):
    results = IVIMResults(ivim_bi_params)
    results.eval_results(results_bi_exp)

    for element in results_bi_exp:
        pixel_idx = element[0]
        assert results.s_0[pixel_idx] == element[1][-1]
        assert results.f[pixel_idx][0] == element[1][2]
        assert results.f[pixel_idx][1] >= 1 - element[1][2]
        assert results.d[pixel_idx][0] == element[1][0]
        assert results.d[pixel_idx][1] == element[1][1]


def test_get_spectrum():
    pass  # TODO: Implement test


def test_save_to_nii(root, ivim_bi_params, results_bi_exp, img):
    file_path = root / "tests" / ".out" / "test"
    results = IVIMResults(ivim_bi_params)
    results.eval_results(results_bi_exp)
    results.save_to_nii(file_path, img)
    assert (file_path.parent / (file_path.stem + "_d.nii.gz")).is_file()
