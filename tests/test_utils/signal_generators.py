"""Signal generators for creating synthetic IVIM and NNLS signals.

Provides clean signal generation without noise for testing purposes.
Use SNRNoiseModel to add noise to generated signals.
"""

import numpy as np
from scipy import signal as scipy_signal


class IVIMSignalGenerator:
    """Generate synthetic IVIM signals for mono-, bi-, and tri-exponential models.
    
    All methods return clean signals without noise. The signal at b=0 represents S0.
    
    Examples:
        >>> gen = IVIMSignalGenerator()
        >>> b_values = np.array([0, 50, 100, 200, 400, 800])
        >>> signal = gen.generate_monoexp(b_values, D=0.001, S0=1000)
        >>> signal_bi = gen.generate_biexp(b_values, f1=0.3, D1=0.001, f2=0.7, D2=0.02, S0=1000)
    """
    
    @staticmethod
    def generate_monoexp(
        b_values: np.ndarray,
        D: float,
        S0: float = 1.0,
    ) -> np.ndarray:
        """Generate mono-exponential signal: S(b) = S0 * exp(-b * D).
        
        Args:
            b_values: Array of b-values
            D: Diffusion coefficient
            S0: Signal at b=0 (default=1.0)
            
        Returns:
            Signal array of same shape as b_values
        """
        return S0 * np.exp(-b_values * D)
    
    @staticmethod
    def generate_biexp(
        b_values: np.ndarray,
        f1: float,
        D1: float,
        D2: float,
        S0: float = 1.0,
    ) -> np.ndarray:
        """Generate bi-exponential signal: S(b) = S0 * (f1*exp(-b*D1) + (1-f1)*exp(-b*D2)).
        
        Args:
            b_values: Array of b-values
            f1: Fraction of first component (typically perfusion, ~0.1-0.3)
            D1: Diffusion coefficient of first component (typically fast/perfusion, ~0.01-0.05)
            D2: Diffusion coefficient of second component (typically slow/tissue, ~0.001-0.003, for fraction 1-f1)
            S0: Signal at b=0 (default=1.0)
            
        Returns:
            Signal array of same shape as b_values
            
        Note:
            Matches BiExpFitModel convention where:
            - f1 pairs with D1 (typically perfusion: small fraction, large D)
            - (1-f1) pairs with D2 (typically tissue: large fraction, small D)
        """
        return S0 * (
            f1 * np.exp(-b_values * D1) + (1 - f1) * np.exp(-b_values * D2)
        )
    
    @staticmethod
    def generate_biexp_reduced(
        b_values: np.ndarray,
        f1: float,
        D1: float,
        D2: float,
    ) -> np.ndarray:
        """Generate bi-exponential signal with reduced parameters: S(b) = f1*exp(-b*D1) + (1-f1)*exp(-b*D2).
        
        Args:
            b_values: Array of b-values
            f1: Fraction of first component (f2 = 1 - f1)
            D1: Diffusion coefficient of first component
            D2: Diffusion coefficient of second component
            
        Returns:
            Signal array of same shape as b_values
        """
        return f1 * np.exp(-b_values * D1) + (1 - f1) * np.exp(-b_values * D2)
    
    @staticmethod
    def generate_triexp(
        b_values: np.ndarray,
        f1: float,
        D1: float,
        f2: float,
        D2: float,
        D3: float,
        S0: float = 1.0,
    ) -> np.ndarray:
        """Generate tri-exponential signal: S(b) = S0 * (f1*exp(-b*D1) + f2*exp(-b*D2) + (1-f1-f2)*exp(-b*D3)).
        
        Args:
            b_values: Array of b-values
            f1: Fraction of first component
            D1: Diffusion coefficient of first component
            f2: Fraction of second component
            D2: Diffusion coefficient of second component
            D3: Diffusion coefficient of third component (f3 = 1 - f1 - f2)
            S0: Signal at b=0 (default=1.0)
            
        Returns:
            Signal array of same shape as b_values
            
        Note:
            Fractions sum to 1, scaled by S0.
        """
        return S0 * (
            f1 * np.exp(-b_values * D1)
            + f2 * np.exp(-b_values * D2)
            + (1 - f1 - f2) * np.exp(-b_values * D3)
        )
    
    @staticmethod
    def generate_triexp_reduced(
        b_values: np.ndarray,
        f1: float,
        D1: float,
        f2: float,
        D2: float,
        D3: float,
    ) -> np.ndarray:
        """Generate tri-exponential signal with reduced parameters: S(b) = f1*exp(-b*D1) + f2*exp(-b*D2) + (1-f1-f2)*exp(-b*D3).
        
        Args:
            b_values: Array of b-values
            f1: Fraction of first component
            D1: Diffusion coefficient of first component
            f2: Fraction of second component
            D2: Diffusion coefficient of second component
            D3: Diffusion coefficient of third component (f3 = 1 - f1 - f2)
            
        Returns:
            Signal array of same shape as b_values
        """
        return (
            f1 * np.exp(-b_values * D1)
            + f2 * np.exp(-b_values * D2)
            + (1 - f1 - f2) * np.exp(-b_values * D3)
        )


class NNLSSignalGenerator:
    """Generate synthetic NNLS signals from diffusion spectra.
    
    Examples:
        >>> gen = NNLSSignalGenerator()
        >>> b_values = np.array([0, 50, 100, 200, 400, 800])
        >>> bins = np.logspace(-4, -1, 100)  # D range from 0.0001 to 0.1
        >>> spectrum = np.zeros(100)
        >>> spectrum[30] = 0.7  # Peak at bins[30]
        >>> spectrum[60] = 0.3  # Peak at bins[60]
        >>> signal = gen.generate_from_spectrum(b_values, spectrum, bins)
    """
    
    @staticmethod
    def generate_from_spectrum(
        b_values: np.ndarray,
        spectrum: np.ndarray,
        bins: np.ndarray,
    ) -> np.ndarray:
        """Generate signal from diffusion spectrum: S(b) = sum_i[spectrum[i] * exp(-b * bins[i])].
        
        Args:
            b_values: Array of b-values
            spectrum: Diffusion spectrum (amplitudes at each bin)
            bins: Diffusion coefficient bins
            
        Returns:
            Signal array of same shape as b_values
        """
        signal = np.zeros_like(b_values, dtype=np.float64)
        for i, d in enumerate(bins):
            signal += spectrum[i] * np.exp(-b_values * d)
        return signal
    
    @staticmethod
    def create_discrete_spectrum(
        bins: np.ndarray,
        d_values: list[float],
        fractions: list[float],
    ) -> np.ndarray:
        """Create a discrete spectrum with unit impulses at specified D values.
        
        Args:
            bins: Diffusion coefficient bins
            d_values: List of D values where peaks should occur
            fractions: List of fractions (peak heights) at each D value
            
        Returns:
            Spectrum array of same shape as bins
            
        Note:
            Each D value is mapped to the nearest bin index.
        """
        spectrum = np.zeros_like(bins, dtype=np.float64)
        
        for d_val, frac in zip(d_values, fractions):
            # Find nearest bin index
            idx = np.argmin(np.abs(bins - d_val))
            spectrum[idx] += frac
        
        return spectrum
    
    @staticmethod
    def generate_multi_component(
        b_values: np.ndarray,
        bins: np.ndarray,
        d_values: list[float],
        fractions: list[float],
    ) -> np.ndarray:
        """Generate multi-component NNLS signal with discrete peaks.
        
        This is a convenience method that combines create_discrete_spectrum
        and generate_from_spectrum.
        
        Args:
            b_values: Array of b-values
            bins: Diffusion coefficient bins
            d_values: List of D values where peaks should occur
            fractions: List of fractions (peak heights) at each D value
            
        Returns:
            Signal array of same shape as b_values
        """
        spectrum = NNLSSignalGenerator.create_discrete_spectrum(bins, d_values, fractions)
        return NNLSSignalGenerator.generate_from_spectrum(b_values, spectrum, bins)
