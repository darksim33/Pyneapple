"""Example: Segmented two-step fitting (illustrative).

This script demonstrates the SegmentedFitter workflow: fit a simple model
(monoexponential) on high b-values to estimate baseline parameters, then
fit a complex model (biexponential) on all b-values with those parameters fixed.

This is a documentation example and not a full benchmark.
"""

from __future__ import annotations

import numpy as np

from pyneapple.fitters import SegmentedFitter
from pyneapple.models import MonoExpModel, BiExpModel
from pyneapple.solvers import CurveFitSolver


def main():
    # Step 1 solver: simple monoexponential for ADC estimation
    solver1 = CurveFitSolver(
        model=MonoExpModel(),
        max_iter=250,
        tol=1e-8,
        p0={"S0": 1000.0, "D": 0.001},
        bounds={"S0": (1.0, 5000.0), "D": (1e-5, 0.1)},
    )

    # Step 2 solver: biexponential with D fixed from step 1
    solver2 = CurveFitSolver(
        model=BiExpModel(fit_s0=False),
        max_iter=500,
        tol=1e-8,
        p0={"f1": 0.2, "D1": 0.001, "D2": 0.02},
        bounds={
            "f1": (0.01, 0.99),
            "D1": (1e-5, 0.003),
            "D2": (0.003, 0.3),
        },
    )

    # Create segmented fitter
    fitter = SegmentedFitter(
        step1_solver=solver1,
        step2_solver=solver2,
        step1_bvalue_range=(200, None),  # b >= 200 for Step 1
        fixed_from_step1=["D"],  # fix D from Step 1 in Step 2
        param_mapping={"D": "D2"},  # MonoExp D → BiExp D2
    )

    # Synthetic data for demonstration
    bvalues = np.array([0, 50, 100, 200, 400, 800], dtype=float)

    # Create synthetic signal: biexponential with D2=0.001, D1=0.01, f1=0.2
    S0 = 1000.0
    D2_true = 0.001
    D1_true = 0.01
    f1_true = 0.2
    signal_true = S0 * (
        f1_true * np.exp(-bvalues * D1_true)
        + (1 - f1_true) * np.exp(-bvalues * D2_true)
    )

    # Tile to create 4D image (4x4 spatial, 6 b-values)
    image = np.tile(signal_true, (4, 4, 1, 1))
    segmentation = np.ones((4, 4, 1), dtype=int)  # all voxels

    # Run the fit
    fitter.fit(bvalues, image, segmentation)
    params = fitter.get_fitted_params()

    print("Segmented fit completed.")
    print("Step 1 params (ADC):", fitter.step1_params_)
    print("Step 2 params:", list(params.keys()))


if __name__ == "__main__":
    main()
