from pyneapple import IVIMResults

from .test_results import save_spectrum_to_excel, save_curve_to_excel


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


def test_save_spectrum_to_excel(ivim_bi_params, array_result, out_excel):
    result = IVIMResults(ivim_bi_params)
    save_spectrum_to_excel(array_result, out_excel, result)


def test_save_curve_to_excel(ivim_bi_params, array_result, out_excel):
    result = IVIMResults(ivim_bi_params)
    save_curve_to_excel(array_result, out_excel, result)


def test_save_to_nii(root, ivim_bi_params, results_bi_exp, img):
    file_path = root / "tests" / ".out" / "test"
    results = IVIMResults(ivim_bi_params)
    results.eval_results(results_bi_exp)
    results.save_to_nii(file_path, img)

    assert (file_path.parent / (file_path.stem + "_d.nii.gz")).is_file()
    assert (file_path.parent / (file_path.stem + "_f.nii.gz")).is_file()
    assert (file_path.parent / (file_path.stem + "_s0.nii.gz")).is_file()
    assert (file_path.parent / (file_path.stem + "_t1.nii.gz")).is_file()

    # TODO: Check if the files are correct

    results.save_to_nii(file_path, img, separate_files=True)
    for idx in range(2):
        assert (file_path.parent / (file_path.stem + f"_d_{idx}.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + f"_f_{idx}.nii.gz")).is_file()
    assert (file_path.parent / (file_path.stem + f"_s0.nii.gz")).is_file()

    for file in file_path.parent.glob("*.nii.gz"):
        file.unlink()


def test_save_to_heatmap(root, ivim_bi_params, results_bi_exp, img):
    file_path = root / "tests" / ".out" / "test"
    results = IVIMResults(ivim_bi_params)
    results.eval_results(results_bi_exp)
    n_slice = 0
    results.save_heatmap(file_path, img, n_slice)

    for idx in range(2):
        assert (file_path.parent / (file_path.stem + f"_{n_slice}_d_{idx}.png")).is_file()
        assert (file_path.parent / (file_path.stem + f"_{n_slice}_f_{idx}.png")).is_file()
    assert (file_path.parent / (file_path.stem + f"_{n_slice}_s_0.png")).is_file()

    for file in file_path.parent.glob("*.png"):
        file.unlink()
