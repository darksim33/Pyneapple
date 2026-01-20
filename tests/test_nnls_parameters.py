"""
Tests for NNLS parameter configuration and validation.

This module tests the NNLS (Non-Negative Least Squares) parameter classes
including initialization, basis matrix generation, argument preparation for
fitting, and serialization to/from JSON and TOML formats.

Key test scenarios:
- Basic NNLS parameters initialization
- NNLS with cross-validation (NNLSCV) parameters
- Basis matrix construction for different configurations
- Pixel-wise and segmentation-wise argument preparation
- Parameter serialization and deserialization
- Different regularization orders (0, 1, 2, 3)
"""
import pytest

from pyneapple import NNLSParams, NNLSCVParams
from pyneapple import Results

from .test_toolbox import ParameterTools


class TestNNLSParameters:
    """Test suite for NNLS parameter initialization and configuration."""

    def test_nnls_init_parameters(self):
        """Test that NNLSParams initializes successfully with default values."""
        assert NNLSParams()

    def test_nnls_get_basis(self, nnls_params):
        """
        Test NNLS basis matrix generation.
        
        Validates that the basis matrix for NNLS fitting is constructed
        correctly with proper dimensions (n_bins + n_bvalues × n_bins)
        and normalized values between 0 and 1.
        """
        basis = nnls_params.get_basis()
        assert basis.shape == (
            nnls_params.boundaries["n_bins"] + nnls_params.b_values.shape[0],
            nnls_params.boundaries["n_bins"],
        )
        assert basis.max() == 1
        assert basis.min() == 0
        assert True

    def test_nnls_get_pixel_args(self, nnls_params, img, seg):
        """
        Test pixel-wise argument preparation for NNLS fitting.
        
        Validates that get_pixel_args correctly prepares fitting arguments
        for pixel-by-pixel NNLS fitting from image and segmentation data.
        """
        args = nnls_params.get_pixel_args(img, seg)
        assert args is not None

    @pytest.mark.parametrize("seg_number", [1, 2, 3])
    def test_nnls_get_seg_args(self, nnls_params, img, seg, seg_number):
        """
        Test segmentation-wise argument preparation for NNLS fitting.
        
        Validates that get_seg_args correctly prepares fitting arguments
        for segment-by-segment NNLS fitting. Tests multiple segment numbers
        to ensure consistency across different segmentation regions.
        """
        args = nnls_params.get_seg_args(img, seg, seg_number)
        assert args is not None

    def test_nnls_json_save(self, nnls_params, out_json):
        """
        Test NNLS parameter serialization to and deserialization from JSON.
        
        Validates that NNLS parameters can be saved to JSON format and
        reloaded without loss of information. Compares all attributes
        between original and reloaded parameters.
        """
        # Test NNLS
        nnls_params.save_to_json(out_json)
        test_params = NNLSParams(out_json)
        attributes = ParameterTools.compare_parameters(nnls_params, test_params)
        ParameterTools.compare_attributes(nnls_params, test_params, attributes)
        assert True

    def test_nnls_load_from_toml(self, nnls_toml_params_file, out_toml):
        """
        Test NNLS parameter loading from TOML and saving with different configurations.
        
        Validates that NNLS parameters can be loaded from TOML format,
        modified (regularization order and mu), saved back to TOML, and
        reloaded correctly. Tests all regularization orders (0-3) to ensure
        complete serialization support.
        """
        nnls_params = NNLSParams(nnls_toml_params_file)
        # Test NNLS with different regularization orders and mu        
        for reg_order in [0, 1, 2, 3]:            
            nnls_params.fit_model.reg_order = reg_order
            mu = reg_order * 0.01 + 0.01
            nnls_params.fit_model.mu = mu
            # Save parameters to TOML file
            nnls_params.save_to_toml(out_toml)
            
            # Load parameters from TOML file
            loaded_params = NNLSParams(out_toml)

            # Compare original and loaded parameters
            attributes = ParameterTools.compare_parameters(nnls_params, loaded_params)
            ParameterTools.compare_attributes(nnls_params, loaded_params, attributes)
            assert loaded_params.fit_model.reg_order == nnls_params.fit_model.reg_order == reg_order
            assert loaded_params.fit_model.mu == nnls_params.fit_model.mu 
            assert True

    # NNLS_CV
    def test_nnls_cv_init_parameters(self):
        """Test that NNLSCVParams initializes successfully with default values."""
        assert NNLSCVParams()

    def test_nnlscv_json_save(self, nnlscv_params, out_json):
        """
        Test NNLS CV parameter serialization to and deserialization from JSON.
        
        Validates that NNLS cross-validation parameters can be saved to JSON
        format and reloaded without loss of information. The CV-specific
        settings (tolerance, etc.) should be preserved correctly.
        """
        # Test NNLS CV
        nnlscv_params.save_to_json(out_json)
        test_params = NNLSCVParams(out_json)
        attributes = ParameterTools.compare_parameters(nnlscv_params, test_params)
        ParameterTools.compare_attributes(nnlscv_params, test_params, attributes)
        assert True
