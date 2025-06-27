""" Module for GPU fitting using pygpufit.
"""

from __future__ import annotations

import numpy as np

from ..utils.logger import logger
from pygpufit import gpufit as gpufit
from .. import IVIMParams, IVIMSegmentedParams


def gpu_fitter(data: zip, params: IVIMParams | IVIMSegmentedParams, **kwargs):
    """
    Fit data using GPU fitting.
    Args:
        data (zip): Zipped data to fit.
        params (IVIMParams | IVIMSegmentedParams): Parameters for fitting.
        **kwargs:
            fit_tolerance (float, optional): Tolerance for fitting.
            parameters_to_fit (np.ndarray, optional): "logical" array of parameters to fit.
            estimator (gpufit.EstimatorID, optional): Estimator for fitting (LSE, MLE).

    Returns:
        list: List of tuples with pixel indices and fit results.
    """

    if not gpufit.cuda_available():
        error_msg = "CUDA not available for GPU fitting."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if isinstance(data, zip):
        pixel_indices, data_list = [], []
        for element in data:
            pixel_indices.append(element[0])
            data_list.append(element[1])
        fit_data = np.array(data_list)
    else:
        error_msg = "Data for GPU fitting must be zipped."
        logger.error(error_msg)
        raise ValueError(error_msg)

    n_parameters = params.n_components * 2  # Number of parameters to fit
    if params.fit_reduced:
        n_parameters -= 1
    if params.fit_t1:
        n_parameters += 1

    fit_model = getattr(gpufit.ModelID, params.model, None)
    if fit_model is None:
        error_msg = "Invalid model for GPU fitting."
        logger.error(error_msg)
        raise ValueError(error_msg)

    start_values = np.tile(
        params.boundaries.start_values.astype(np.float32),
        (fit_data.shape[0], 1),
    )

    constraints = np.tile(
        np.float32(
            list(
                zip(
                    params.boundaries.lower_bounds,
                    params.boundaries.upper_bounds,
                )
            )
        ).flatten(),
        (fit_data.shape[0], 1),
    )
    constraint_types = np.squeeze(
        np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (n_parameters, 1))
    )
    b_values = np.squeeze(params.b_values).astype(np.float32)

    tolerance = getattr(kwargs, "fit_tolerance", params.fit_tolerance)
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
