"""Example: Pixelwise fitting using a TOML config (illustrative only).

This script demonstrates the minimal steps to load a TOML config, build the
configured fitter, and run a fit on an in-memory synthetic dataset. It is a
documentation example and not intended to be a full test or benchmark.
"""

from __future__ import annotations

import numpy as np

from pyneapple.models import BiExpModel
from pyneapple.solvers import CurveFitSolver
from pyneapple.fitters import PixelWiseFitter


def main():
    solver = CurveFitSolver(
        model=BiExpModel(fit_s0=True),
        max_iter=500,
        tol=1e-8,
        p0={"S0": 1000.0, "f1": 0.2, "D1": 0.01, "D2": 0.001},
        bounds={
            "S0": (1.0, 5000.0),
            "f1": (0.01, 0.99),
            "D1": (1e-5, 1e-3),
            "D2": (1e-3, 0.1),
        },
    )

    fitter = PixelWiseFitter(solver=solver)
    bvalues = np.array(
        [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200],
        dtype=float,
    )
    # Create synthetic signal: biexponential with D2=0.001, D1=0.01, f1=0.2
    S0 = 1000.0
    D2_true = 0.001
    D1_true = 0.01
    f1_true = 0.2
    signal_true = S0 * (
        f1_true * np.exp(-bvalues * D1_true)
        + (1 - f1_true) * np.exp(-bvalues * D2_true)
    )
    # Tile to create 4D image (4x4 spatial, 16 b-values)
    image = np.tile(signal_true, (4, 4, 1, 1))
    segmentation = np.ones((4, 4, 1), dtype=int)  # all voxels
    fitter.fit(bvalues, image, segmentation=segmentation)
    params = fitter.get_fitted_params()
    print("Fitted parameter keys:", list(params.keys()))  # type: ignore
    print("Fitted parameter shapes:", {k: v.shape for k, v in params.items()})  # type: ignore


if __name__ == "__main__":
    main()
