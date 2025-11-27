from unittest.mock import Mock

import matplotlib
import numpy as np
import pytest

from pyneapple import IVIMResults, IVIMSegmentedParams, IVIMSegmentedResults
from radimgarray import SegImgArray

matplotlib.use("Agg")


class TestIVIMResults:
    def test_eval_results(self, ivim_bi_params, results_bi_exp):
        results = IVIMResults(ivim_bi_params)
        results.eval_results(results_bi_exp)

        for element in results_bi_exp:
            pixel_idx = element[0]
            assert results.S0[pixel_idx] == np.sum([element[1][0], element[1][2]])
            assert results.f[pixel_idx][0] == element[1][0] / results.S0[pixel_idx]
            assert results.f[pixel_idx][1] == element[1][2] / results.S0[pixel_idx]
            assert results.D[pixel_idx][0] == element[1][1]
            assert results.D[pixel_idx][1] == element[1][3]

    def test_get_spectrum(self, ivim_bi_params):
        results = IVIMResults(ivim_bi_params)
        bins = results._get_bins(101, (0.1, 1.0))
        d_value_indexes = [np.random.randint(1, 50), np.random.randint(51, 101)]
        d_values = [float(bins[d_value_indexes[0]]), float(bins[d_value_indexes[1]])]
        f = np.random.random()
        fractions = [f, 1 - f]
        # s_0_values = [np.random.randint(1, 2500)]
        test_result = [
            (
                (0, 0, 0),
                np.array([fractions[0], d_values[0], fractions[1], d_values[1]]),
            )
        ]
        results.eval_results(test_result)
        results.get_spectrum(
            101,
            (0.1, 1.0),
        )
        assert fractions[0] == results.spectrum[(0, 0, 0)][d_value_indexes[0]]
        assert 1 - fractions[0] == results.spectrum[(0, 0, 0)][d_value_indexes[1]]

    def test_save_to_nii(self, temp_dir, ivim_bi_params, results_bi_exp, img):
        file_path = temp_dir / "test"
        results = IVIMResults(ivim_bi_params)
        results.eval_results(results_bi_exp)

        results.save_to_nii(file_path, img)
        assert (file_path.parent / (file_path.stem + "_d.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + "_f.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + "_s0.nii.gz")).is_file()
        # assert (file_path.parent / (file_path.stem + "_t1.nii.gz")).is_file()

        results.save_to_nii(file_path, img, separate_files=True)
        for idx in range(2):
            assert (file_path.parent / (file_path.stem + f"_d_{idx}.nii.gz")).is_file()
            assert (file_path.parent / (file_path.stem + f"_f_{idx}.nii.gz")).is_file()
        assert (file_path.parent / (file_path.stem + "_s0.nii.gz")).is_file()

        for file in file_path.parent.glob("*.nii.gz"):
            file.unlink()

    def test_save_to_heatmap(self, root, ivim_bi_params, results_bi_exp, img):
        file_path = root / "tests" / ".temp" / "test"
        results = IVIMResults(ivim_bi_params)
        results.eval_results(results_bi_exp)
        n_slice = 0
        results.save_heatmap(file_path, img, n_slice)

        for idx in range(2):
            assert (
                file_path.parent / (file_path.stem + f"_{n_slice}_d_{idx}.png")
            ).is_file()
            assert (
                file_path.parent / (file_path.stem + f"_{n_slice}_f_{idx}.png")
            ).is_file()
        assert (file_path.parent / (file_path.stem + f"_{n_slice}_s_0.png")).is_file()

        for file in file_path.parent.glob("*.png"):
            file.unlink()


class TestIVIMSegmentedResults:
    # ---  Unit tests for IVIMSegmentedResults

    @pytest.fixture
    def mock_params(self):
        """Create a mock IVIMSegmentedParams object."""
        params = Mock(spec=IVIMSegmentedParams)
        params.fixed_component = "D_1"
        params.fixed_t1 = False

        # Mock the fit_model
        fit_model = Mock()
        fit_model.args = ["f_1", "D_1", "f_2", "D_2", "S_0"]  # BiExp with S0
        fit_model.fit_t1 = False
        fit_model.fit_reduced = False
        params.fit_model = fit_model

        return params

    @pytest.fixture
    def mock_params_with_t1(self):
        """Create a mock IVIMSegmentedParams object with T1 fitting."""
        params = Mock(spec=IVIMSegmentedParams)
        params.fixed_component = "D_1"
        params.fixed_t1 = True

        # Mock the fit_model
        fit_model = Mock()
        fit_model.args = [
            "f_1",
            "D_1",
            "f_2",
            "D_2",
            "S_0",
            "T_1",
        ]  # BiExp with S0 and T1
        fit_model.fit_t1 = True
        fit_model.fit_reduced = False
        params.fit_model = fit_model

        return params

    @pytest.fixture
    def sample_results(self):
        """Create sample fitting results (without fixed component)."""
        # Results for a BiExp model with D1 fixed, so we have: f1, f2, D2, S0
        return [
            ((0, 0, 0), np.array([0.15, 0.85, 0.003, 1000.0])),  # f1, f2, D2, S0
            ((0, 0, 1), np.array([0.25, 0.75, 0.004, 1200.0])),  # f1, f2, D2, S0
            ((1, 1, 0), np.array([0.20, 0.80, 0.0035, 1100.0])),  # f1, f2, D2, S0
        ]

    @pytest.fixture
    def sample_fixed_components(self):
        """Create sample fixed components."""
        return [
            {  # D values
                (0, 0, 0): 0.001,
                (0, 0, 1): 0.0012,
                (1, 1, 0): 0.0011,
            },
            {  # T1 values (if needed)
                (0, 0, 0): 800.0,
                (0, 0, 1): 850.0,
                (1, 1, 0): 820.0,
            },
        ]

    def test_add_fixed_components_basic(
        self, mock_params, sample_results, sample_fixed_components
    ):
        """Test basic functionality of add_fixed_components method."""
        results = IVIMSegmentedResults(mock_params)

        # Call add_fixed_components
        modified_results = results.add_fixed_components(
            sample_results, sample_fixed_components
        )

        # Check that we get the same number of results
        assert len(modified_results) == len(sample_results)

        # Check that fixed D1 is inserted at position 1 (after f1, before f2)
        for i, (original, modified) in enumerate(zip(sample_results, modified_results)):
            pixel_coords = original[0]
            original_array = original[1]
            modified_coords, modified_array = modified

            # Coordinates should be unchanged
            assert modified_coords == pixel_coords

            # Array should be one element longer
            assert len(modified_array) == len(original_array) + 1

            # Check that D1 is at the correct position (index 1)
            expected_d1 = sample_fixed_components[0][pixel_coords]
            assert modified_array[1] == expected_d1

            # Check that other values are preserved
            assert modified_array[0] == original_array[0]  # f1
            assert modified_array[2] == original_array[1]  # f2 (shifted)
            assert modified_array[3] == original_array[2]  # D2 (shifted)
            assert modified_array[4] == original_array[3]  # S0 (shifted)

    def test_add_fixed_components_with_t1(
        self, mock_params_with_t1, sample_results, sample_fixed_components
    ):
        """Test add_fixed_components with T1 fitting enabled."""
        results = IVIMSegmentedResults(mock_params_with_t1)

        # Call add_fixed_components
        modified_results = results.add_fixed_components(
            sample_results, sample_fixed_components
        )

        # Check first result in detail
        pixel_coords, modified_array = modified_results[0]
        original_array = sample_results[0][1]

        # Array should be two elements longer (D1 + T1)
        assert len(modified_array) == len(original_array) + 2

        # Check positions
        expected_d1 = sample_fixed_components[0][pixel_coords]
        expected_t1 = sample_fixed_components[1][pixel_coords]

        assert modified_array[1] == expected_d1  # D1 at position 1
        assert modified_array[5] == expected_t1  # T1 at position 5 (last)

    def test_add_fixed_components_none_input(self, mock_params):
        """Test add_fixed_components with None input."""
        results = IVIMSegmentedResults(mock_params)
        sample_results = [((0, 0, 0), np.array([1, 2, 3]))]

        # Should return original results unchanged
        modified_results = results.add_fixed_components(sample_results, None)
        assert modified_results == sample_results

    def test_add_fixed_components_invalid_fixed_d(
        self, mock_params, sample_results, sample_fixed_components
    ):
        """Test add_fixed_components with invalid fixed component name."""
        mock_params.fixed_component = "D5"  # Not in args
        results = IVIMSegmentedResults(mock_params)

        # Should return original results unchanged
        modified_results = results.add_fixed_components(
            sample_results, sample_fixed_components
        )
        assert modified_results == sample_results

    def test_eval_results_integration(
        self, mock_params, sample_results, sample_fixed_components
    ):
        """Test that eval_results correctly calls parent after adding fixed components."""
        results = IVIMSegmentedResults(mock_params)

        # Mock the parent eval_results method
        original_parent_eval = IVIMSegmentedResults.__bases__[0].eval_results
        parent_eval_mock = Mock()
        IVIMSegmentedResults.__bases__[0].eval_results = parent_eval_mock

        try:
            # Call eval_results
            results.eval_results(
                sample_results, fixed_component=sample_fixed_components
            )

            # Check that parent eval_results was called with modified results
            parent_eval_mock.assert_called_once()
            call_args = parent_eval_mock.call_args
            modified_results = call_args[0][0]  # First positional argument

            # Check that the modified results have the correct structure
            assert len(modified_results) == len(sample_results)

            # Check that first result has the fixed component inserted
            pixel_coords, modified_array = modified_results[0]
            expected_d1 = sample_fixed_components[0][pixel_coords]
            assert modified_array[1] == expected_d1

        finally:
            # Restore original method
            IVIMSegmentedResults.__bases__[0].eval_results = original_parent_eval

    def test_different_fixed_positions_unit(self):
        """Test with different fixed component positions."""
        # Test with D2 fixed instead of D1
        params = Mock(spec=IVIMSegmentedParams)
        params.fixed_component = "D2"
        params.fixed_t1 = False

        fit_model = Mock()
        fit_model.args = ["f1", "D1", "f2", "D2", "S0"]
        fit_model.fit_t1 = False
        params.fit_model = fit_model

        results = IVIMSegmentedResults(params)

        sample_results = [
            ((0, 0, 0), np.array([0.15, 0.001, 0.85, 1000.0]))
        ]  # f1, D1, f2, S0
        fixed_components = [{(0, 0, 0): 0.003}]  # D2 value

        modified_results = results.add_fixed_components(
            sample_results, fixed_components
        )

        # D2 should be inserted at position 3 (after f2)
        pixel_coords, modified_array = modified_results[0]
        assert modified_array[3] == 0.003  # D2 fixed value
        assert len(modified_array) == 5  # Original 4 + 1 fixed

    # ---  Integration tests for IVIMSegmentedResults

    def test_complete_workflow(self):
        """Test the complete workflow of IVIMSegmentedResults."""

        # Create a simple mock parameters object
        class MockFitModel:
            def __init__(self):
                self.args = ["f1", "D1", "f2", "D2", "S0"]
                self.fit_t1 = False
                self.fit_reduced = False
                self.n_components = 2

            def model(self, b_values, *args, **kwargs):
                # Simple mock model that returns zeros
                return np.zeros_like(b_values)

        class MockParams:
            def __init__(self):
                self.fixed_component = "D1"
                self.fixed_t1 = False
                self.fit_model = MockFitModel()
                self.b_values = np.array([0, 50, 100, 200, 400, 800])

        # Create the results object
        params = MockParams()
        results = IVIMSegmentedResults(params)

        # Simulate fitting results (without the fixed D1 component)
        # These would come from the second fitting step
        fitting_results = [
            ((0, 0, 0), np.array([0.15, 0.85, 0.003, 1000.0])),  # f1, f2, D2, S0
            ((1, 1, 0), np.array([0.20, 0.80, 0.0035, 1100.0])),  # f1, f2, D2, S0
        ]

        # Fixed components from the first fitting step
        fixed_components = [
            {  # D1 values for each pixel
                (0, 0, 0): 0.001,
                (1, 1, 0): 0.0012,
            }
        ]

        # Test the add_fixed_components method directly
        modified_results = results.add_fixed_components(
            fitting_results, fixed_components
        )

        # Verify the structure
        assert len(modified_results) == 2

        # Check first pixel
        pixel_coords, params_array = modified_results[0]
        assert pixel_coords == (0, 0, 0)
        assert len(params_array) == 5  # f1, D1, f2, D2, S0

        # Verify the order: f1, D1 (fixed), f2, D2, S0
        assert params_array[0] == 0.15  # f1
        assert params_array[1] == 0.001  # D1 (fixed)
        assert params_array[2] == 0.85  # f2
        assert params_array[3] == 0.003  # D2
        assert params_array[4] == 1000.0  # S0

        # Check second pixel
        pixel_coords, params_array = modified_results[1]
        assert pixel_coords == (1, 1, 0)
        assert params_array[1] == 0.0012  # D1 (fixed) for second pixel

        print("✓ Complete workflow test passed!")
        print("✓ Fixed components correctly inserted at the right positions")
        print("✓ Original parameter order preserved after insertion")

    def test_different_fixed_positions(self):
        """Test with different fixed component positions to verify flexibility."""

        class MockFitModel:
            def __init__(self, args):
                self.args = args
                self.fit_t1 = False
                self.fit_reduced = False
                self.n_components = 2

            def model(self, b_values, *args, **kwargs):
                return np.zeros_like(b_values)

        class MockParams:
            def __init__(self, fixed_component, args):
                self.fixed_component = fixed_component
                self.fixed_t1 = False
                self.fit_model = MockFitModel(args)
                self.b_values = np.array([0, 50, 100])

        # Test Case 1: Fix D2 (second diffusion component)
        params1 = MockParams("D2", ["f1", "D1", "f2", "D2", "S0"])
        results1 = IVIMSegmentedResults(params1)

        # Fitting results without D2
        fitting_results1 = [
            ((0, 0, 0), np.array([0.15, 0.001, 0.85, 1000.0]))
        ]  # f1, D1, f2, S0
        fixed_components1 = [{(0, 0, 0): 0.003}]  # D2 value

        modified1 = results1.add_fixed_components(fitting_results1, fixed_components1)
        pixel_coords, params_array = modified1[0]

        # D2 should be at position 3
        assert params_array[3] == 0.003  # D2 (fixed)
        assert len(params_array) == 5

        # Test Case 2: Fix D1 (first diffusion component) in TriExp model
        params2 = MockParams("D1", ["f1", "D1", "f2", "D2", "f3", "D3"])
        results2 = IVIMSegmentedResults(params2)

        # Fitting results without D1
        fitting_results2 = [
            ((0, 0, 0), np.array([0.1, 0.2, 0.002, 0.7, 0.004]))
        ]  # f1, f2, D2, f3, D3
        fixed_components2 = [{(0, 0, 0): 0.001}]  # D1 value

        modified2 = results2.add_fixed_components(fitting_results2, fixed_components2)
        pixel_coords, params_array = modified2[0]

        # D1 should be at position 1
        assert params_array[1] == 0.001  # D1 (fixed)
        assert len(params_array) == 6

    @pytest.fixture
    def results_bi_exp_fixed(self, seg: SegImgArray):
        shape = np.squeeze(seg).shape
        d_fast_map = np.zeros(shape)
        d_fast_map[np.squeeze(seg) > 0] = np.random.random() * 10**-3
        f_fast_map = np.zeros(shape)
        f_fast_map[np.squeeze(seg) > 0] = np.random.randint(1, 2500)
        f_slow_map = np.zeros(shape)
        f_slow_map[np.squeeze(seg) > 0] = np.random.randint(1, 2500)
        results = []
        for idx in list(zip(*np.where(np.squeeze(seg) > 0))):
            # results.append((idx, np.array([d_fast_map[idx], f_map[idx], s_0_map[idx]])))
            results.append(
                (idx, np.array([f_slow_map[idx], f_fast_map[idx], d_fast_map[idx]]))
            )
        return results

    def test_eval_results(
        self, ivim_bi_t1_params_file, results_bi_exp_fixed, fixed_values
    ):
        params = IVIMSegmentedParams(
            ivim_bi_t1_params_file,
        )
        params.fixed_component = "D_1"
        params.fixed_t1 = True
        params.reduced_b_values = [0, 50, 550, 650]
        params.set_up()
        result = IVIMSegmentedResults(params)
        result.eval_results(results_bi_exp_fixed, fixed_component=fixed_values)
        for element in results_bi_exp_fixed:
            pixel_idx = element[0]
            assert result.S0[pixel_idx] == element[1][0] + element[1][1]
            assert result.f[pixel_idx][0] == element[1][0] / result.S0[pixel_idx]
            assert result.f[pixel_idx][1] == element[1][1] / result.S0[pixel_idx]
            assert result.D[pixel_idx][0] == fixed_values[0][pixel_idx]  # D1 is fixed
            assert result.D[pixel_idx][1] == element[1][2]  # D2 is fitted
            assert result.t1[pixel_idx] == fixed_values[1][pixel_idx]
