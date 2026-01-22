"""Canonical parameter sets for testing IVIM and NNLS models.

Defines standard parameter configurations that represent typical biological scenarios.
These parameters are used across multiple tests to ensure consistency.

Kidney-specific parameters are based on:
    Jasse, J., Wittsack, H.-J., Thiel, T. A., Zukovs, R., Valentin, B., Antoch, G., & Ljimani, A. (2024).
    "Toward Optimal Fitting Parameters for Multi-Exponential DWI Image Analysis of the Human Kidney: 
    A Simulation Study Comparing Different Fitting Algorithms."
    Mathematics, 12(4), 609. https://doi.org/10.3390/math12040609
"""

import numpy as np
from typing import TypedDict


# Standard b-values for IVIM fitting (kidney-specific protocol)
# Using 16 values optimized for kidney compartment differentiation
STANDARD_B_VALUES = np.array(
    [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 525, 750],
    dtype=np.float64,
)

# Minimal b-values for quick testing (less accurate fitting)
# UNUSED - kept for potential future use in quick smoke tests
MINIMAL_B_VALUES = np.array([0, 50, 100, 200, 400, 800], dtype=np.float64)


# ============================================================================
# Parameter ranges for synthetic data generation and fitting bounds
# ============================================================================

# Blood compartment (fast diffusion)
BLOOD_FRACTION_RANGE = (0.08, 0.15)  # 8-15% typical blood volume fraction
BLOOD_DIFFUSION_RANGE = (0.100, 0.200)  # mm²/s, kidney blood/perfusion diffusion

# Tubule compartment (intermediate diffusion)
TUBULE_FRACTION_RANGE = (0.25, 0.35)  # 25-35% tubule volume fraction
TUBULE_DIFFUSION_RANGE = (0.004, 0.008)  # mm²/s, kidney tubular diffusion

# Tissue compartment (slow diffusion)
TISSUE_DIFFUSION_RANGE = (0.0008, 0.0015)  # mm²/s, kidney tissue/parenchyma diffusion

# Effective tissue+tubule diffusion (for bi-exponential models)
TISSUE_COMBINED_RANGE = (0.001, 0.003)  # mm²/s, combined tissue+tubule

# Signal intensity
S0_RANGE = (180, 220)  # Typical signal at b=0 for synthetic data

# Fitting bounds (wider than typical ranges to allow optimizer freedom)
FITTING_BOUNDS = {
    "mono": {
        "S0": (10, 2500),
        "D": (0.0005, 0.005),
    },
    "biexp": {
        "S0": (10, 2500),
        "f1": (0.02, 0.30),  # Allow 2-30% blood fraction for fitting
        "D1": (0.050, 0.250),  # Allow 0.05-0.25 mm²/s blood diffusion
        "D2": (0.0005, 0.010),  # Allow 0.0005-0.01 mm²/s tissue diffusion
    },
    "triexp": {
        "S0": (10, 2500),
        "f1": (0.03, 0.25),  # Blood fraction
        "D1": (0.050, 0.250),  # Blood diffusion
        "f2": (0.15, 0.50),  # Tubule fraction
        "D2": (0.002, 0.015),  # Tubule diffusion
        "D3": (0.0005, 0.005),  # Tissue diffusion
    },
}


class IVIMParameters(TypedDict, total=False):
    """Type definition for IVIM parameter dictionaries."""
    S0: float
    D: float
    f1: float
    D1: float
    f2: float
    D2: float
    f3: float
    D3: float
    description: str


# ============================================================================
# Mono-exponential parameters
# ============================================================================

MONO_TYPICAL: IVIMParameters = {
    "S0": 1000.0,
    "D": 0.001,  # Kidney tissue diffusion coefficient (d_slow)
    "description": "Kidney mono-exponential diffusion (tissue ADC)",
}


# ============================================================================
# Bi-exponential parameters
# ============================================================================

BIEXP_TYPICAL: IVIMParameters = {
    "S0": 1000.0,
    "f1": 0.10,  # 10% blood fraction (f_fast)
    "D1": 0.165,  # Blood diffusion coefficient (d_fast)
    "D2": 0.002,  # Combined tissue/tubule diffusion (effective d_slow, for 1-f1=0.9)
    # Note: Model is S0 * (f1*exp(-b*D1) + (1-f1)*exp(-b*D2))
    # Kidney-specific: f1=0.10 (blood) with D1=0.165, (1-f1)=0.90 (tissue+tubule) with D2≈0.002
    "description": "Kidney bi-exponential IVIM (10% blood, 90% tissue+tubule)",
}

BIEXP_LOW_PERFUSION: IVIMParameters = {
    "S0": 1000.0,
    "f1": 0.05,  # 5% blood fraction (low perfusion kidney)
    "D1": 0.150,  # Blood diffusion coefficient
    "D2": 0.0015,  # Tissue diffusion coefficient (for 1-f1 = 95%)
    # Note: Model is S0 * (f1*exp(-b*D1) + (1-f1)*exp(-b*D2))
    "description": "Low perfusion kidney bi-exponential IVIM (5% blood, 95% tissue)",
}

BIEXP_HIGH_PERFUSION: IVIMParameters = {
    "S0": 1000.0,
    "f1": 0.20,  # 20% blood fraction (high perfusion kidney)
    "D1": 0.180,  # Blood diffusion coefficient (higher for highly vascularized)
    "D2": 0.0025,  # Tissue diffusion coefficient (for 1-f1 = 80%)
    # Note: Model is S0 * (f1*exp(-b*D1) + (1-f1)*exp(-b*D2))
    "description": "High perfusion kidney bi-exponential IVIM (20% blood, 80% tissue)",
}


# ============================================================================
# Tri-exponential parameters
# ============================================================================

TRIEXP_TYPICAL: IVIMParameters = {
    "S0": 1000.0,
    "f1": 0.10,  # 10% blood fraction (f_fast)
    "D1": 0.165,  # Blood diffusion coefficient (d_fast)
    "f2": 0.30,  # 30% tubule fraction (f_inter)
    "D2": 0.0058,  # Tubule diffusion coefficient (d_inter)
    # Note: f3 = 1 - f1 - f2 = 0.60 (60% tissue)
    "D3": 0.001,  # Tissue diffusion coefficient (d_slow)
    "description": "Kidney tri-exponential IVIM (10% blood + 30% tubule + 60% tissue)",
}

# UNUSED - kept for potential future test scenarios
TRIEXP_COMPLEX: IVIMParameters = {
    "S0": 1000.0,
    "f1": 0.15,  # 15% blood fraction (elevated perfusion)
    "D1": 0.180,  # Blood diffusion coefficient (higher)
    "f2": 0.25,  # 25% tubule fraction
    "D2": 0.0050,  # Tubule diffusion coefficient
    # Note: f3 = 1 - f1 - f2 = 0.60 (60% tissue)
    "D3": 0.0012,  # Tissue diffusion coefficient
    "description": "Complex kidney tri-exponential IVIM (15% blood + 25% tubule + 60% tissue)",
}


# ============================================================================
# NNLS parameters (UNUSED - NNLSSignalGenerator removed)
# ============================================================================
# These parameters are retained for potential future NNLS synthetic signal tests

class NNLSParameters(TypedDict):
    """Type definition for NNLS parameter dictionaries."""
    d_range: tuple[float, float]
    n_bins: int
    d_values: list[float]
    fractions: list[float]
    description: str


NNLS_BIMODAL: NNLSParameters = {
    "d_range": (0.0007, 0.3),  # Kidney NNLS range: D_min=0.7×10^-3, D_max=300×10^-3
    "n_bins": 300,  # M=300 from kidney study
    "d_values": [0.002, 0.165],  # Tissue/tubule and blood components
    "fractions": [0.90, 0.10],  # 90% tissue+tubule, 10% blood
    "description": "Kidney NNLS bi-modal spectrum (tissue+tubule + blood)",
}

NNLS_TRIMODAL: NNLSParameters = {
    "d_range": (0.0007, 0.3),  # Kidney NNLS range
    "n_bins": 300,  # M=300 from kidney study
    "d_values": [0.001, 0.0058, 0.165],  # Tissue, tubule, blood
    "fractions": [0.60, 0.30, 0.10],  # Kidney three-compartment model
    "description": "Kidney NNLS tri-modal spectrum (tissue + tubule + blood)",
}


# ============================================================================
# Default testing parameters
# ============================================================================

# Default SNR for synthetic data generation (kidney-specific)
# SNR=140 represents high-quality kidney DWI imaging
# This matches the standard simulation parameters from kidney IVIM studies
DEFAULT_SNR = 140.0

# Default random seed for reproducibility
DEFAULT_SEED = 42
