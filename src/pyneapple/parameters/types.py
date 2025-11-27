"""
This module defines the `Parameters` type, which is a union of various parameter classes used in the fitting process.

Type Aliases:
    Parameters: A union of IVIMParams, IVIMSegmentedParams, NNLSParams, and NNLSCVParams.
"""

from typing import Union

from .ideal import IDEALParams
from .ivim import IVIMParams, IVIMSegmentedParams
from .nnls import NNLSCVParams, NNLSParams
from .parameters import BaseParams

Parameters = Union[
    BaseParams, IVIMParams, IVIMSegmentedParams, IDEALParams, NNLSParams, NNLSCVParams
]
