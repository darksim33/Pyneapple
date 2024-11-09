import pytest
import numpy as np
from radimgarray import SegImgArray

from pyneapple import IVIMResults, IVIMSegmentedResults, IVIMSegmentedParams

from .test_toolbox import ResultTools as Tools


class TestIVIMResults:
    def test_eval_results(self, ivim_bi_params, results_bi_exp):
        results = IVIMResults(ivim_bi_params)
        results.eval_results(results_bi_exp)

        for element in results_bi_exp:
            pixel_idx = element[0]
            assert results.s_0[pixel_idx] == element[1][-1]
            assert results.f[pixel_idx][0] == element[1][2]
            assert results.f[pixel_idx][1] >= 1 - element[1][2]
            assert results.d[pixel_idx][0] == element[1][0]
            assert results.d[pixel_idx][1] == element[1][1]

    def test_get_spectrum(self, ivim_bi_params):
        results = IVIMResults(ivim_bi_params)
        bins = results._get_bins(101, (0.1, 1.0))
        d_value_indexes = [np.random.randint(1, 50), np.random.randint(51, 101)]
        d_values = [float(bins[d_value_indexes[0]]), float(bins[d_value_indexes[1]])]
        fractions = [np.random.random()]
        s_0_values = [np.random.randint(1, 2500)]
        test_result = [((0, 0, 0), np.array(d_values + fractions + s_0_values))]
        results.eval_results(test_result)
        results.get_spectrum(
            101,
            (0.1, 1.0),
        )
        assert fractions[0] == results.spectrum[(0, 0, 0)][d_value_indexes[0]]
        assert 1 - fractions[0] == results.spectrum[(0, 0, 0)][d_value_indexes[1]]

    def test_save_spectrum_to_excel(self, ivim_bi_params, array_result, out_excel):
        result = IVIMResults(ivim_bi_params)
        Tools.save_spectrum_to_excel(array_result, out_excel, result)

    def test_save_curve_to_excel(self, ivim_bi_params, array_result, out_excel):
        result = IVIMResults(ivim_bi_params)
        Tools.save_curve_to_excel(array_result, out_excel, result)

    def test_save_to_nii(self, root, ivim_bi_params, results_bi_exp, img):
        file_path = root / "tests" / ".out" / "test"
        results = IVIMResults(ivim_bi_params)
        results.eval_results(results_bi_exp)
        
        results.save_to_nii(file_path, img)
        assert (file_path.parent / (file_path.stem + "_d.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + "_f.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + "_s0.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + "_t1.nii.gz")).is_file()

        results.save_to_nii(file_path, img, separate_files=True)
        for idx in range(2):
            assert (file_path.parent / (file_path.stem + f"_d_{idx}.nii.gz")).is_file()
            assert (file_path.parent / (file_path.stem + f"_f_{idx}.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + f"_s0.nii.gz")).is_file()

        for file in file_path.parent.glob("*.nii.gz"):
            file.unlink()

    def test_save_to_heatmap(self, root, ivim_bi_params, results_bi_exp, img):
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


class TestIVIMSegmentedResults:

    @pytest.fixture
    def results_bi_exp_fixed(self, seg: SegImgArray):
        shape = np.squeeze(seg).shape
        d_fast_map = np.zeros(shape)
        d_fast_map[np.squeeze(seg) > 0] = np.random.random() * 10 ** -3
        f_map = np.zeros(shape)
        f_map[np.squeeze(seg) > 0] = np.random.random()
        s_0_map = np.zeros(shape)
        s_0_map[np.squeeze(seg) > 0] = np.random.randint(1, 2500)
        results = []
        for idx in list(zip(*np.where(np.squeeze(seg) > 0))):
            results.append((idx, np.array([d_fast_map[idx], f_map[idx], s_0_map[idx]])))
        return results

    def test_eval_results(self, ivim_bi_params_file, results_bi_exp_fixed, fixed_values):
        params = IVIMSegmentedParams(
            ivim_bi_params_file,
            fixed_component="D_slow",
            fixed_t1=True,
            reduced_b_values=[0, 50, 550, 650],
        )
        result = IVIMSegmentedResults(params)
        result.eval_results(results_bi_exp_fixed, fixed_component=fixed_values)
        for element in results_bi_exp_fixed:
            pixel_idx = element[0]
            assert result.s_0[pixel_idx] == element[1][-1]
            assert result.f[pixel_idx][0] == element[1][1]
            assert result.f[pixel_idx][1] >= 1 - element[1][1]
            assert result.d[pixel_idx][1] == element[1][0]
            assert result.d[pixel_idx][0] == fixed_values[0][pixel_idx]
            assert result.t_1[pixel_idx] == fixed_values[1][pixel_idx]
