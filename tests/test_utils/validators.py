"""Validators for parameter recovery in fitting tests.

Provides SNR-dependent tolerance checking for fitted parameters.
"""

import numpy as np
from typing import Union, Dict, Any


def get_tolerance_from_snr(snr: float) -> float:
    """Calculate relative tolerance based on SNR.
    
    Uses a simple stepped mapping tuned for IVIM parameter recovery:
    - SNR >= 40: 10% tolerance (high quality data, realistic fitting variability)
    - 30 <= SNR < 40: 15% tolerance (good quality data)
    - 20 <= SNR < 30: 20% tolerance (moderate quality data)
    - SNR < 20: 25% tolerance (low quality data)
    
    Note: Even with high SNR, IVIM parameter estimation has inherent
    uncertainty due to model ill-conditioning and parameter correlation.
    
    Args:
        snr: Signal-to-noise ratio
        
    Returns:
        Relative tolerance (rtol) as a fraction (e.g., 0.10 for 10%)
    """
    if snr >= 40:
        return 0.10  # 10%
    elif snr >= 30:
        return 0.15  # 15%
    elif snr >= 20:
        return 0.20  # 20%
    else:
        return 0.25  # 25%


def validate_parameter_recovery(
    fitted: Union[float, np.ndarray, Dict[str, float]],
    expected: Union[float, np.ndarray, Dict[str, float]],
    snr: float,
    param_name: str = "parameter",
    custom_tolerance: float | None = None,
) -> None:
    """Validate that fitted parameters match expected values within SNR-dependent tolerance.
    
    Args:
        fitted: Fitted parameter value(s) - can be scalar, array, or dict
        expected: Expected (ground truth) parameter value(s) - same type as fitted
        snr: Signal-to-noise ratio used to determine tolerance
        param_name: Name of parameter for error messages (default="parameter")
        custom_tolerance: Override SNR-based tolerance with custom value (optional)
        
    Raises:
        AssertionError: If fitted values don't match expected within tolerance
        
    Examples:
        >>> # Single parameter
        >>> validate_parameter_recovery(0.0305, 0.03, snr=30, param_name="f1")
        
        >>> # Multiple parameters as array
        >>> fitted = np.array([0.295, 0.00098, 0.021])
        >>> expected = np.array([0.30, 0.001, 0.020])
        >>> validate_parameter_recovery(fitted, expected, snr=30, param_name="IVIM_params")
        
        >>> # Parameters as dictionary
        >>> fitted = {"f1": 0.295, "D1": 0.00098}
        >>> expected = {"f1": 0.30, "D1": 0.001}
        >>> validate_parameter_recovery(fitted, expected, snr=30)
    """
    # Determine tolerance
    rtol = custom_tolerance if custom_tolerance is not None else get_tolerance_from_snr(snr)
    
    # Handle dictionary inputs
    if isinstance(fitted, dict) and isinstance(expected, dict):
        for key in expected:
            if key not in fitted:
                raise AssertionError(
                    f"Parameter '{key}' missing from fitted results. "
                    f"Available keys: {list(fitted.keys())}"
                )
            
            fitted_val = fitted[key]
            expected_val = expected[key]
            
            np.testing.assert_allclose(
                fitted_val,
                expected_val,
                rtol=rtol,
                atol=1e-10,
                err_msg=(
                    f"Parameter recovery failed for '{key}': "
                    f"fitted={fitted_val:.6f}, expected={expected_val:.6f}, "
                    f"tolerance={rtol*100:.1f}% (SNR={snr})"
                ),
            )
        return
    
    # Handle scalar/array inputs
    # Convert to arrays to ensure type consistency for assert_allclose
    fitted_arr = np.asarray(fitted)
    expected_arr = np.asarray(expected)
    
    np.testing.assert_allclose(
        fitted_arr,
        expected_arr,
        rtol=rtol,
        atol=1e-10,
        err_msg=(
            f"Parameter recovery failed for '{param_name}': "
            f"fitted={fitted}, expected={expected}, "
            f"tolerance={rtol*100:.1f}% (SNR={snr})"
        ),
    )


def validate_fraction_sum(
    fractions: Union[list, np.ndarray],
    expected_sum: float = 1.0,
    rtol: float = 1e-10,
) -> None:
    """Validate that fractions sum to expected value (typically 1.0).
    
    Args:
        fractions: Array or list of fraction values
        expected_sum: Expected sum of fractions (default=1.0)
        rtol: Relative tolerance for sum check (default=1e-10)
        
    Raises:
        AssertionError: If sum of fractions doesn't match expected value
    """
    fractions_array = np.atleast_1d(fractions)
    actual_sum = np.sum(fractions_array)
    
    np.testing.assert_allclose(
        actual_sum,
        expected_sum,
        rtol=rtol,
        atol=1e-10,
        err_msg=(
            f"Fraction sum validation failed: "
            f"sum={actual_sum:.10f}, expected={expected_sum:.10f}, "
            f"fractions={fractions_array}"
        ),
    )


def validate_signal_decay(
    signal: np.ndarray,
    param_name: str = "signal",
) -> None:
    """Validate that signal exhibits monotonic decay (first value >= last value).
    
    This is a basic sanity check for diffusion-weighted signals.
    
    Args:
        signal: Signal array (should decay from b=0 to max b-value)
        param_name: Name for error messages (default="signal")
        
    Raises:
        AssertionError: If signal does not decay monotonically
    """
    if signal[0] < signal[-1]:
        raise AssertionError(
            f"{param_name} should decay from b=0 to max b-value: "
            f"S(b=0)={signal[0]:.3f}, S(b_max)={signal[-1]:.3f}"
        )


def validate_model_consistency(
    fitted_curve: np.ndarray,
    recalculated_curve: np.ndarray,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Validate that stored fitted curve matches model recalculation.
    
    This ensures that the fitted parameters, when passed back to the model,
    reproduce the stored fitted curve.
    
    Args:
        fitted_curve: Curve stored in fitting results
        recalculated_curve: Curve recalculated from fitted parameters
        rtol: Relative tolerance (default=1e-12)
        atol: Absolute tolerance (default=1e-12)
        
    Raises:
        AssertionError: If curves don't match within tolerance
    """
    np.testing.assert_allclose(
        fitted_curve,
        recalculated_curve,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"Model consistency check failed: "
            f"Stored curve and recalculated curve from fitted parameters don't match. "
            f"max_diff={np.max(np.abs(fitted_curve - recalculated_curve)):.2e}"
        ),
    )
