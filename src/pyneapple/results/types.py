"""
This module defines the type `Results` which is a union of all results types.

Type Aliases:
    Results (Union[IVIMResults, IVIMSegmentedResults, NNLSResults]): A union of all results types.
"""

from typing import Union
from .ivim import IVIMResults, IVIMSegmentedResults
from .nnls import NNLSResults

Results = Union[IVIMResults, IVIMSegmentedResults, NNLSResults]
