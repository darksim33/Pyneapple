"""
This file is used for debugging the visual studio GPUfit workflow.

"""
from pathlib import Path
import numpy as np
import pygpufit.gpufit as gpufit
from pyneapple import IVIMParams
from tests.conftest import decay_tri

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    file = root / r"tests/.data/fitting/default_params_IVIM_tri.json"
    params = IVIMParams(file)
    fit_data = decay_tri(params)
    fit_args = fit_data["fit_args"]
    n_fits = fit_args.shape[0]
    starts = [210, 0.001, 210, 0.02, 210, 0.01]
    lower = [10, 0.0007, 10, 0.003, 10, 0.003]
    upper = [2500, 0.05, 2500, 0.05, 2500, 0.3]

    start_values = np.tile(np.float32(starts), (n_fits, 1))
    constraints = np.tile(
        np.float32(list(zip(lower, upper))).flatten(),
        (n_fits, 1),
    )
    b_values = np.tile(np.char.array(params.b_values.T), (n_fits, 1))
    constraint_types = np.squeeze(
        np.tile(np.int32(gpufit.ConstraintType.LOWER_UPPER), (6, 1))
    )

    result = gpufit.fit_constrained(
        fit_args,
        None,
        13,
        initial_parameters=start_values,
        constraints=constraints,
        constraint_types=constraint_types,
        tolerance=0.000001,
        max_number_iterations=250,
        parameters_to_fit=None,
        estimator_id=gpufit.EstimatorID.LSE,
        user_info=b_values,
    )