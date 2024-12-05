""" Module for GPU fitting using pygpufit.
"""

from __future__ import annotations

import numpy as np

from pygpufit import gpufit as gpufit
from .. import IVIMParams, IVIMSegmentedParams


def reorder_array(array: np.ndarray) -> np.ndarray:
    """Adjust Oder to fit GPU fitting Models.
    From D1,D2,...F1,F2,... to F1,D1,F2,D2,... for GPU fitting.
    """
    n = len(array)
    if n % 2 != 0:
        raise ValueError("Array length must be even.")
    reordered = []
    half = n // 2
    for i in range(half):
        reordered.append(array[half + i])
        reordered.append(array[i])
    return np.array(reordered)


def gpu_fitter(data: zip, params: IVIMParams | IVIMSegmentedParams, **kwargs):
    """
    Fit data using GPU fitting.
    Args:
        data (zip): Zipped data to fit.
        params (IVIMParams | IVIMSegmentedParams): Parameters for fitting.
        **kwargs:
            tolerance (float): Tolerance for fitting.
            parameters_to_fit (np.ndarray): "logical" array of parameters to fit.
            estimator (gpufit.EstimatorID): Estimator for fitting (LSE, MLE).

    Returns:
        list: List of tuples with pixel indices and fit results.
    """

    if isinstance(data, zip):
        pixel_indices, data_list = [], []
        for element in data:
            pixel_indices.append(element[0])
            data_list.append(element[1])
        # pixel_indices = [element[0] for element in data]
        # data_list = [element[1] for element in data]
        fit_data = np.array(data_list)
    else:
        raise ValueError("Data for GPU fitting must be zipped.")

    n_parameters = params.n_components * 2  # Number of parameters to fit
    if params.n_components == 1:
        fit_model = "MONO_EXP"
    elif params.n_components == 2:
        fit_model = "BI_EXP"
    elif params.n_components == 3:
        fit_model = "TRI_EXP"
    else:
        raise ValueError("Invalid number of components for GPU fitting.")
    if params.scale_image == "S/S0":
        fit_model += "_RED"
        n_parameters -= 1

    fit_model = getattr(gpufit.ModelID, fit_model, None)
    if fit_model is None:
        raise ValueError("Invalid model for GPU fitting.")

    start_values = np.tile(
        reorder_array(params.boundaries.start_values.astype(np.float32)),
        (fit_data.shape[0], 1),
    )

    constraints = np.tile(
        np.float32(
            list(
                zip(
                    reorder_array(params.boundaries.lower_stop_values),
                    reorder_array(params.boundaries.upper_stop_values),
                )
            )
        ).flatten(),
        (fit_data.shape[0], 1),
    )
    constraint_types = np.squeeze(
        np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (n_parameters, 1))
    )
    b_values = np.squeeze(params.b_values).astype(np.float32)

    tolerance = getattr(kwargs, "tolerance", 1e-6)  # TODO: Add parameter for tolerance
    parameters_to_fit = getattr(kwargs, "parameters_to_fit", None)
    estimator = getattr(kwargs, "estimator", gpufit.EstimatorID.LSE)

    result = gpufit.fit_constrained(
        fit_data,
        None,
        fit_model,
        initial_parameters=start_values,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=tolerance,
        max_number_iterations=params.max_iter,
        parameters_to_fit=parameters_to_fit,  # NOTE: What happens if the number of parameters is reduced?
        estimator_id=estimator,
        user_info=b_values,
    )
    fit_results = [
        (pixel_indices[i], result[0][i, :]) for i in range(len(pixel_indices))
    ]

    return fit_results  # NOTE: export fit quality parameters
