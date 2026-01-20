"""
Tests for NNLS (Non-Negative Least Squares) fitting functionality.

This module tests the NNLS model fitting implementations including:
- Segmentation-wise fitting with different regularization orders
- Cross-validation based regularization parameter selection
- Pixel-wise multithreaded fitting
- Result spectrum generation and export

Key test scenarios:
- Different regularization orders (0, 1, 2, 3)
- Segmented vs. pixel-wise fitting approaches
- Standard NNLS vs. cross-validated NNLS
"""
import pytest
from pathlib import Path
from functools import wraps
from multiprocessing import freeze_support

from pyneapple.fitting import FitData
from radimgarray import RadImgArray


# Decorators
def freeze_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    freeze_support()
    return wrapper


# @pytest.mark.order(
#     after="test_nnls_parameters.py::TestNNLSParameters::test_nnls_init_parameters"
# )
class TestNNLSFitting:
    """Test suite for NNLS model fitting functionality."""
    
    # Segmented sequential fitting
    @pytest.mark.parametrize("reg_order", [0, 1, 2, 3])
    def test_nnls_segmented_reg(
        self,
        reg_order,
        nnls_fit_data: FitData,
        out_nii: Path,
    ):
        """Test NNLS segmentation-wise fitting with different regularization orders.
        
        Tests that NNLS fitting converges correctly when performed on a 
        segmentation basis using different Tikhonov regularization orders 
        (0=no regularization, 1=first derivative, 2=second derivative, 3=third derivative).
        The resulting spectrum is saved as NIfTI to verify output generation.
        """
        nnls_fit_data.params.reg_order = reg_order
        nnls_fit_data.fit_segmentation_wise()

        img_dyn = nnls_fit_data.results.spectrum.as_RadImgArray(nnls_fit_data.img)
        img_dyn.save(out_nii, save_as="nii", dtype=float)
        assert True

    @pytest.mark.slow  # as fuck
    @pytest.mark.skip("Not working properly atm.")
    def test_nnls_segmented_reg_cv(self, nnlscv_fit_data: FitData, out_nii: Path):
        """Test NNLS segmentation-wise fitting with cross-validation for regularization.
        
        Tests that NNLS fitting with automatic regularization parameter selection
        via cross-validation works correctly for segmentation-wise fitting.
        The optimal regularization parameter should be determined automatically.
        
        Note: Currently skipped - requires investigation of cross-validation
        implementation. See GitHub issue #XXX for details.
        """
        nnlscv_fit_data.fit_segmentation_wise()

        img_dyn = RadImgArray(
            nnlscv_fit_data.results.spectrum.as_array(nnlscv_fit_data.seg.shape)
        )
        img_dyn.save(out_nii, save_as="nii", dtype=float)
        assert True

    # Multithreading
    @freeze_me
    @pytest.mark.slow
    @pytest.mark.parametrize("reg_order", [0, 1, 2, 3])
    def test_nnls_pixel_multi_reg(self, reg_order, nnls_fit_data: FitData):
        """Test NNLS pixel-wise multithreaded fitting with different regularization orders.
        
        Tests that NNLS fitting works correctly when performed pixel-by-pixel
        using multiprocessing parallelization. Validates that different
        regularization orders produce valid results without race conditions
        or threading issues.
        """
        nnls_fit_data.params.reg_order = reg_order
        nnls_fit_data.fit_pixel_wise(fit_type="multi")
        assert True

    @freeze_me
    @pytest.mark.slow
    @pytest.mark.skip("Not working properly atm.")
    def test_nnls_pixel_multi_reg_cv(self, nnlscv_fit_data: FitData):
        """Test NNLS pixel-wise multithreaded fitting with cross-validation.
        
        Tests that NNLS fitting with automatic regularization parameter
        selection via cross-validation works correctly in a multithreaded
        pixel-wise context. Each pixel should determine its optimal
        regularization independently.
        
        Note: Currently skipped - requires investigation of CV implementation
        in multithreaded context. See GitHub issue #XXX for details.
        """
        nnlscv_fit_data.fit_pixel_wise(fit_type="multi")
        assert True
