"""Test utilities for creating synthetic signals and validating fitting results.

This module provides:
- Signal generators for IVIM models
- SNR-based noise models
- Canonical parameter sets for testing
- Parameter recovery validators
"""

from .signal_generators import IVIMSignalGenerator
from .noise_models import SNRNoiseModel
from .validators import validate_parameter_recovery

__all__ = [
    "IVIMSignalGenerator",
    "SNRNoiseModel",
    "validate_parameter_recovery",
]
