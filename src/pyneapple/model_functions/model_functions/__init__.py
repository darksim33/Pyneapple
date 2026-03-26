"""Pure stateless forward model functions.

This module provides the mathematical core of PyNeapple's signal models
as simple, stateless functions. They can be used directly or as building
blocks for Model and Solver classes.

Submodules:
    multiexp — Mono/bi/tri-exponential forward functions and T1 modifiers
    nnls     — Basis matrix, regularization, and signal reconstruction
"""

from .multiexp import (
    apply_t1,
    apply_t1_steam,
    biexp_forward,
    biexp_reduced_forward,
    biexp_s0_forward,
    monoexp_forward,
    monoexp_reduced_forward,
    triexp_forward,
    triexp_reduced_forward,
    triexp_s0_forward,
)
from .nnls import (
    build_regularized_basis,
    curvature_matrix,
    get_basis,
    get_bins,
    reconstruct_signal,
    regularization_matrix,
)

__all__ = [
    # IVIM exponential functions
    "monoexp_forward",
    "monoexp_reduced_forward",
    "biexp_forward",
    "biexp_reduced_forward",
    "biexp_s0_forward",
    "triexp_forward",
    "triexp_reduced_forward",
    "triexp_s0_forward",
    "apply_t1",
    "apply_t1_steam",
    # NNLS functions
    "get_bins",
    "get_basis",
    "regularization_matrix",
    "curvature_matrix",
    "reconstruct_signal",
    "build_regularized_basis",
]
