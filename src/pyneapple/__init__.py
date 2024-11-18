from typing import Union

from .models import IVIM, NNLS, NNLSCV

from .parameters.types import Parameters
from .parameters.ivim import IVIMParams, IVIMSegmentedParams
from .parameters.nnls import NNLSParams, NNLSCVParams
from .parameters.ideal import IDEALParams

from .results.types import Results
from .results.ivim import IVIMResults, IVIMSegmentedResults
from .results.nnls import NNLSResults
from .fitting.fitdata import FitData
