from functools import partial
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
            elif attr in ["fit_model", "fit_function"]:
                continue
            else:
                assert getattr(params1, attr) == getattr(params2, attr)


class ResultTools:
    @staticmethod
    def compare_lists_of_floats(list_1: list, list_2: list):
        """Compares lists of floats"""
        list_1 = [round(element, 10) for element in list_1]
        list_2 = [round(element, 10) for element in list_2]
        assert list_1 == list_2

    @staticmethod
    def save_spectrum_to_excel(array_result, out_excel, result):
        for idx in np.ndindex(array_result.shape[:-2]):
            spectrum = array_result[idx]
            result.spectrum.update({idx: spectrum})

        # if out_excel.is_file():
        #     out_excel.unlink()
        bins = np.linspace(0, 10, 11)
        result.save_spectrum_to_excel(out_excel, bins)
        df = pd.read_excel(out_excel, index_col=0)
        columns = df.columns.tolist()
        assert columns == ["pixel"] + bins.tolist()
        for idx, key in enumerate(result.spectrum.keys()):
            spectrum = np.array(df.iloc[idx, 1:])
            ResultTools.compare_lists_of_floats(
                spectrum.tolist(), np.squeeze(result.spectrum[key]).tolist()
            )

    @staticmethod
    def save_curve_to_excel(array_result, out_excel, result):
        for idx in np.ndindex(array_result.shape[:-2]):
            curve = array_result[idx]
            result.curve.update({idx: curve})

        b_values = np.linspace(0, 10, 11).tolist()
        result.save_fit_curve_to_excel(out_excel, b_values)
        df = pd.read_excel(out_excel, index_col=0)
        columns = df.columns.tolist()
        assert columns == ["pixel"] + b_values
        for idx, key in enumerate(result.curve.keys()):
            curve = np.array(df.iloc[idx, 1:])
            ResultTools.compare_lists_of_floats(
                curve.tolist(), np.squeeze(result.curve[key].tolist())
            )
