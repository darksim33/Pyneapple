from __future__ import annotations

import numpy as np

from pygpufit import gpufit as gf
from .. import IVIMParams, IVIMSegmentedParams


def gpu_fitter(data: zip, params: IVIMParams | IVIMSegmentedParams):

    if isinstance(data, zip):
        data_list = list()
        for data_tuple in data:
            data_list.append(data_tuple[1])
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
    if params.scale_images == "S/S0":
        fit_model += "_RED"
        n_parameters -= 1

    start_values = np.zeros((fit_data.shape[0], params.n_components))
    constraints = np.zeros((fit_data.shape[0], params.n_components * 2))
    constraint_types = np.zeros((n_parameters, 1))
    b_values = np.zeros((fit_data.shape[0], params.b_values.shape[0]))
    for n_fit in range(fit_data.shape[0]):
        start_values[n_fit, :] = params.boundaries.start_values
        constraints[n_fit, 0::2] = params.boundaries.lower_bounds
        constraints[n_fit, 1::2] = params.boundaries.upper_bounds
        constraint_types[n_fit] = gf.ConstraintType.LOWER_UPPER
        b_values[n_fit, :] = params.b_values

    tolerance = 1e-6  # TODO: Add parameter for tolerance

    result = gf.fit_constrained(
        fit_data,
        None,
        fit_model,
        initial_parameters=start_values,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=tolerance,
        max_number_iterations=params.max_iter,
        parameters_to_fit=np.int32(n_parameters),
        estimator_id=gf.EstimatorID.LSE,
        user_info=b_values,
    )
