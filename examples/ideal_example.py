"""Example: IDEAL iterative multi-resolution fitting (illustrative).

This script demonstrates the IDEALFitter workflow: start with a coarse grid,
fit the model, then interpolate parameters to progressively finer grids
and refit. This coarse-to-fine approach improves convergence and reduces
local minima.

This is a documentation example and not a full benchmark.
"""

from __future__ import annotations

import numpy as np

from pyneapple.fitters import IDEALFitter
from pyneapple.models import BiExpModel
from pyneapple.solvers import CurveFitSolver


def main():
    # Configure solver with biexponential model
    solver = CurveFitSolver(
        model=BiExpModel(fit_s0=False),
        max_iter=250,
        tol=1e-8,
        p0={"f1": 0.2, "D1": 0.001, "D2": 0.02},
        bounds={
            "f1": (0.01, 0.99),
            "D1": (1e-5, 0.003),
            "D2": (0.003, 0.3),
        },
    )

    # Create IDEAL fitter with 4 resolution levels
    # Starting from 16x16, going to 32x32, 64x64, then full 128x128
    # New format: each row is one step (n_steps, ideal_dims)
    fitter = IDEALFitter(
        solver=solver,
        dim_steps=np.array([[16, 16], [32, 32], [64, 64], [128, 128]]),
        step_tol=[0.5, 0.2, 0.2, 0.2],
        ideal_dims=2,
        segmentation_threshold=0.2,
        interpolation_method="cubic",
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

    # Tile to create 4D image (128x128 spatial, 6 b-values)
    # Must match the final dim_steps column
    image = np.tile(signal_true, (128, 128, 1, 1))
    segmentation = np.ones((128, 128, 1), dtype=int)  # all voxels

    # Run the IDEAL fit
    fitter.fit(bvalues, image, segmentation)
    params = fitter.get_fitted_params()

    print("IDEAL fit completed.")
    print("Parameter keys:", list(params.keys()))
    print(f"D2 range: [{params['D2'].min():.4f}, {params['D2'].max():.4f}]")
    print(f"D1 range: [{params['D1'].min():.4f}, {params['D1'].max():.4f}]")
    print(f"f1 range: [{params['f1'].min():.4f}, {params['f1'].max():.4f}]")


if __name__ == "__main__":
    main()
