import math
import numpy as np
from scipy.optimize import least_squares


def model_exp_wrapper(bvalues: np.array, signal: np.array, np: int):
    # JJasse.2022
    #
    def multi_exp(x):
        model = np.array(0)
        for i in range(np - 2):
            model = model + np.array(math.exp(-np.kron(bvalues, abs(x(i)))) * x(np + i))
        print(model)
        return (
            model
            + np.array(
                math.exp(-np.kron(bvalues, abs(x(np - 1)))) * (100 - (np.sum(x[np:-1])))
            )
            - signal
        )

    return multi_exp


def model_mono_t1(bvalues: np.array, signal: np.array, TM: int):
    # x[0] = S0; x[1] = D; x[2] = T1
    # model = np.array(0)
    def model(x):
        return (
            x[0] * math.exp(-np.kron(bvalues, abs(x[1]))) * math.exp(x[2] / TM) - signal
        )

    return model


def fit_adc(
    bvalues: np.array,
    signal: np.array,
    mixingtime: int,
    lb: np.array | None = None,  # lower Bounds
    ub: np.array | None = None,  # upper Bounds
    startingValues: np.array | None = None,
):
    if not lb:
        lb = np.array([10, 0.0001, 1000])
    if not ub:
        ub = np.array([1000, 0.01, 2500])
    if not startingValues:
        startingValues = np.array([50, 0.001, 1750])

    results = least_squares(
        model_mono_t1(bvalues, signal, mixingtime),
        startingValues,
        bounds=(lb, ub),
        method="lm",
    )

    return results


bvalues = np.array([0, 100, 150, 750])
signal = np.array([100, 75, 50, 10])

test = fit_adc(bvalues, signal, 50)
