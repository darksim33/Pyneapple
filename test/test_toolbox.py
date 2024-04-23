from functools import partial
import numpy as np


class ParameterTools(object):
    @staticmethod
    def get_attributes(item) -> list:
        return [
            attr
            for attr in dir(item)
            if not callable(getattr(item, attr))
            and not attr.startswith("_")
            and not isinstance(getattr(item, attr), partial)
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

        assert attributes == test_attributes
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
            elif attr == "boundaries":
                ParameterTools.compare_boundaries(
                    getattr(params1, attr), getattr(params2, attr)
                )
            else:
                assert getattr(params1, attr) == getattr(params2, attr)
