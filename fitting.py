import math
import numpy as np
from scipy.optimize import least_squares, curve_fit
from typing import Callable


class fitParameters:
    def __init__(
        self,
        bvalues,
        fitfunction: Callable,
        lb: np.ndarray | None = None,  # lower bound
        ub: np.ndarray | None = None,  # upper bound
        x0: np.ndarray | None = None,  # starting values
        TM: np.ndarray | None = None,  # mixing time
    ):
        self.bvalues = bvalues
        self.fitfunction = fitfunction
        self.lb = lb if lb is not None else np.array([10, 0.0001, 1000])
        self.ub = ub if ub is not None else np.array([1000, 0.01, 2500])
        self.x0 = x0 if x0 is not None else np.array([50, 0.001, 1750])
        self.TM = TM

    def load_bvals(self, file: str):
        with open(file, "r") as f:
            self.bvalues = np.array([int(x) for x in f.read().split(" ")])


def model_mono_t1(TM: int):
    def model(bvalues: np.ndarray, S0, D, T1):
        return np.array(S0 * np.exp(-np.kron(bvalues, D)) * np.exp(-T1 / TM))

    return model


def model_multi_exp(nComponents: int):
    def model(bValues: np.ndarray, X: np.ndarray):
        function = np.array()
        for ii in range(
            nComponents - 2
        ):  # for 1 component the idx gets negative and for is evaded
            function = function + np.array(
                np.exp(-np.kron(bValues, abs(X[ii + 1]) * X[nComponents + ii + 1]))
            )
        return X[0] * (
            function
            + np.array(
                np.exp(-np.kron(bValues, abs(X[nComponents])))
                * (1 - np.sum(X[nComponents + 1 : -1]))
            )
        )

    return model


def fit_adc(signal: np.ndarray, fitparams: fitParameters) -> None:
    results = curve_fit(
        fitparams.fitfunction,
        fitparams.bvalues,
        signal,
        fitparams.x0,
        bounds=(fitparams.lb, fitparams.ub),
    )

    return results


bvalues = np.array([0, 100, 150, 750])
signal = np.array([100, 75, 50, 10])

fitParams = fitParameters(bvalues, model_mono_t1(50), TM=50)
test = fit_adc(signal, fitParams)
print(test[0])
