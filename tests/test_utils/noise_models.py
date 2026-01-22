"""SNR-based noise model for adding realistic noise to synthetic signals.

Noise is calculated based on Signal-to-Noise Ratio (SNR) at b=0.
"""

import numpy as np


class SNRNoiseModel:
    """Add Gaussian noise to signals based on SNR.
    
    The noise standard deviation is calculated as: sigma = S0 / SNR
    where S0 is the signal intensity at b=0 (first element of signal array).
    
    Typical IVIM SNR values:
        - Clinical data: SNR = 20-50
        - High quality: SNR = 40-60
        - Low quality: SNR = 10-20
    
    Examples:
        >>> noise_model = SNRNoiseModel()
        >>> clean_signal = np.array([1000, 900, 800, 700])
        >>> noisy_signal = noise_model.add_noise(clean_signal, snr=30, seed=42)
        >>> # sigma = 1000 / 30 = 33.33
    """
    
    @staticmethod
    def add_noise(
        signal: np.ndarray,
        snr: float,
        seed: int | None = None,
    ) -> np.ndarray:
        """Add Gaussian noise to signal based on SNR.
        
        Args:
            signal: Clean signal array (first element should be S0 at b=0)
            snr: Signal-to-noise ratio (positive float)
            seed: Random seed for reproducibility (default=None)
            
        Returns:
            Noisy signal array of same shape as input
            
        Raises:
            ValueError: If SNR <= 0 or signal is empty
        """
        if snr <= 0:
            raise ValueError(f"SNR must be positive, got {snr}")
        
        if signal.size == 0:
            raise ValueError("Signal array is empty")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate noise standard deviation from S0 (signal at b=0)
        s0 = signal.flat[0]  # First element (b=0)
        sigma = s0 / snr
        
        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, size=signal.shape)
        
        return signal + noise
    
    # calculate_sigma() and calculate_snr() methods removed - were unused in test suite
    # SNR/sigma calculations are done inline where needed
