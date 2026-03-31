"""Example: NNLS fitting example (illustrative).

Shows how to load a NNLS config and run a simple in-memory fit. This is
intended as a short example for the docs and not a complete benchmark.
"""

from __future__ import annotations

import numpy as np

from pyneapple.models import NNLSModel
from pyneapple.solvers import NNLSSolver
from pyneapple.fitters import PixelWiseFitter


def main():
    solver = NNLSSolver(
        model=NNLSModel(d_range=(0.0007, 0.5), n_bins=250),
        max_iter=500,
        tol=1e-8,
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
