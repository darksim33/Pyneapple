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


# NNLSSignalGenerator class removed - was unused in test suite
# If NNLS synthetic signal tests are needed in the future, this can be re-added
