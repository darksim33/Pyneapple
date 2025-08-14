"""
This module defines the `Parameters` type, which is a union of various parameter classes used in the fitting process.

Type Aliases:
    Parameters: A union of IVIMParams, IVIMSegmentedParams, NNLSParams, and NNLSCVParams.
"""

from typing import Union

from .ivim import IVIMParams, IVIMSegmentedParams
from .nnls import NNLSParams, NNLSCVParams
from .ideal import IDEALParams

Parameters = Union[
    IVIMParams, IVIMSegmentedParams, IDEALParams, NNLSParams, NNLSCVParams
]
