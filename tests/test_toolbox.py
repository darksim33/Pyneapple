from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from pyneapple import IVIMParams


class ParameterTools(object):
    @staticmethod
    def get_attributes(item) -> list:
        return [
            attr
            for attr in dir(item)
            if not callable(getattr(item, attr))
            and not attr.startswith("_")
            and not isinstance(getattr(item, attr), partial)
            and not attr in ["fit_model", "fit_function", "params_1", "params_2"]
        ]

    @staticmethod
    def compare_boundaries(boundary1, boundary2):
        attributes1 = ParameterTools.get_attributes(boundary1)
        attributes2 = ParameterTools.get_attributes(boundary2)

        assert attributes1 == attributes2

        for attr in attributes1:
            if isinstance(getattr(boundary1, attr), np.ndarray):
                assert getattr(boundary1, attr).all() == getattr(boundary2, attr).all()
            else:
                assert getattr(boundary1, attr) == getattr(boundary2, attr)

    @staticmethod
    def compare_parameters(params1, params2) -> list:
        """
        Compares two parameter sets.

        Args:
            params1: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
            params2: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams

        Returns:

        """
        # compare attributes first
        attributes = ParameterTools.get_attributes(params1)
        test_attributes = ParameterTools.get_attributes(params2)

        # Atleast all original parameters should be present in the test parameters
        assert set(attributes).issubset(set(test_attributes))
        return attributes

    @staticmethod
    def compare_attributes(params1, params2, attributes: list):
        """
        Compares attribute values of two parameter sets
        Args:
            params1: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
            params2: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
            attributes:

        Returns:

        """
        for attr in attributes:
            if isinstance(getattr(params1, attr), np.ndarray):
                assert getattr(params1, attr).all() == getattr(params2, attr).all()
            elif attr.lower() == "boundaries":
                ParameterTools.compare_boundaries(
                    getattr(params1, attr), getattr(params2, attr)
                )
            elif attr in ["fit_model", "fit_function"]:
                continue
            elif attr in ["params_1", "params_2"]:  # Special case for SegmentedIVIM
                continue
            else:
                assert getattr(params1, attr) == getattr(params2, attr)
