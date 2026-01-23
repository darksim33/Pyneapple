"""Tests for IDEAL (Iterative Decomposition of water and fat with Echo Asymmetry and Least-squares estimation) fitting.

This module tests IDEAL fitting functionality for pyramidal constraint estimation in MRI:

- Basic fitting: iterative constraint estimation
- Parameter configuration: IDEALParams initialization and usage
- Integration: Working with RadImgArray and result structures

IDEAL fitting estimates IVIM fitting constraints iteratively over multiple resolution steps, refining parameter estimates
at each level. This test suite validates the core fitting logic and parameter handling.
"""
import pytest
import numpy as np
from pyneapple.fitting.fit import fit_ideal
from pyneapple import IDEALParams
from radimgarray import RadImgArray, SegImgArray


@pytest.mark.ideal
class TestIDEALFitting:
    def test_fit_ideal_basic_functionality(self, ideal_params_file, mocker):
        """Test basic functionality of the fit_ideal method."""

        # Create mock data
        img_data = np.random.rand(64, 64, 3, 6).astype(np.float32)
        seg_data = np.ones((64, 64, 3, 1), dtype=np.float32)
        img = RadImgArray(img_data)
        seg = SegImgArray(seg_data)

        # Load parameters
        params = IDEALParams(ideal_params_file)

        # Create mock for fit_handler
        mock_results = [
            (coords, np.random.rand(4))
            for coords in [(i, j, 0) for i in range(8) for j in range(8)]
        ]
        mock_fit_handler = mocker.patch(
            "pyneapple.fitting.fit.fit_handler", return_value=mock_results
        )
        mocker.patch.object(
            params,
            "sort_fit_results",
            return_value=np.random.rand(8, 8, 1, 4).astype(np.float32),
        )

        # Execute function
        results = fit_ideal(img, seg, params)

        # Check that results were returned
        assert results is not None
        # Check that fit_handler was called for each step
        assert mock_fit_handler.call_count == len(params.dim_steps)

    def test_fit_ideal_with_multiple_steps(self, ideal_params_file, mocker):
        """Test fit_ideal mit mehreren Schritten."""
        # Mock-Daten erstellen
        img_data = np.random.rand(64, 64, 3, 6).astype(np.float32)
        seg_data = np.ones((64, 64, 3, 1), dtype=np.float32)
        img = RadImgArray(img_data)
        seg = SegImgArray(seg_data)

        # Parameter laden
        params = IDEALParams(ideal_params_file)
        params.dim_steps = [[8, 8], [16, 16], [32, 32], [64, 64]]  # Explizit setzen

        # Mock für get_boundaries
        mock_bounds = (np.ones((4,)), np.zeros((4,)), np.ones((4,)) * 2)
        mocker.patch.object(params, "get_boundaries", return_value=mock_bounds)

        # Mock für interpolate_img und interpolate_seg
        def mock_interpolate(array, step_idx, **kwargs):
            dims = params.dim_steps[step_idx]
            mock_array = np.ones((dims[0], dims[1], 1, array.shape[3])).astype(
                np.float32
            )
            return RadImgArray(mock_array)

        mocker.patch.object(params, "interpolate_img", side_effect=mock_interpolate)
        mocker.patch.object(params, "interpolate_seg", side_effect=mock_interpolate)

        # Mock für fit_handler
        mock_step_results = [
            (coords, np.random.rand(4))
            for coords in [(i, j, 0) for i in range(8) for j in range(8)]
        ]
        mock_fit_handler = mocker.patch(
            "pyneapple.fitting.fit.fit_handler", return_value=mock_step_results
        )

        # Mock für sort_fit_results
        def mock_sort(img, results):
            shape = (img.shape[0], img.shape[1], img.shape[2], 4)
            return np.ones(shape).astype(np.float32)

        mocker.patch.object(params, "sort_fit_results", side_effect=mock_sort)

        # Funktion ausführen
        results = fit_ideal(img, seg, params)

        # Überprüfen der Ergebnisse
        assert results is not None

        # Überprüfen, dass get_boundaries für jeden Schritt aufgerufen wurde
        assert params.get_boundaries.call_count == len(params.dim_steps)

        # Überprüfen, dass interpolate_img und interpolate_seg für jeden Schritt aufgerufen wurden
        assert params.interpolate_img.call_count == len(params.dim_steps)
        assert params.interpolate_seg.call_count == len(params.dim_steps)

        # Überprüfen, dass fit_handler korrekt aufgerufen wurde
        assert mock_fit_handler.call_count == len(params.dim_steps)

    @pytest.mark.gpu
    def test_fit_ideal_with_different_fit_types(self, ideal_params_file, mocker):
        """Test fit_ideal with different fit types."""
        from pyneapple.fitting.fit import fit_ideal, fit_handler

        # Create mock data
        img_data = np.random.rand(64, 64, 3, 6).astype(np.float32)
        seg_data = np.ones((64, 64, 3, 1), dtype=np.float32)
        img = RadImgArray(img_data)
        seg = SegImgArray(seg_data)

        # Load parameters
        params = IDEALParams(ideal_params_file)
        params.dim_steps = np.array([[8, 8]])  # Only one step for faster tests

        # Mocks for all required methods
        mocker.patch.object(
            params,
            "get_boundaries",
            return_value=(np.ones((4,)), np.zeros((4,)), np.ones((4,)) * 2),
        )
        mocker.patch.object(
            params,
            "interpolate_img",
            return_value=RadImgArray(np.ones((8, 8, 1, 6)).astype(np.float32)),
        )
        mocker.patch.object(
            params,
            "interpolate_seg",
            return_value=RadImgArray(np.ones((8, 8, 1, 1)).astype(np.float32)),
        )
        mocker.patch.object(
            params, "get_pixel_args", return_value=zip([], [], [], [], [])
        )
        mocker.patch("pyneapple.fitting.fit.fit_handler", return_value=[])
        mocker.patch.object(
            params,
            "sort_fit_results",
            return_value=np.ones((8, 8, 1, 4)).astype(np.float32),
        )

        # Test different fit types
        for fit_type in ["single", "multi", "gpu"]:
            results = fit_ideal(img, seg, params, fit_type=fit_type)
            assert results is not None
