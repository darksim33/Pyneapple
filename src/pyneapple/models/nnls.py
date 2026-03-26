"""NNLS distribution model for diffusion MRI."""

import numpy as np
from loguru import logger

from ..model_functions.nnls import get_basis, get_bins
from .base import DistributionModel


class NNLSModel(DistributionModel):
    """Distribution model for NNLS fitting over a log-spaced diffusion grid.

    Represents a continuous distribution of diffusion coefficients as a
    weighted sum of mono-exponential decays on logarithmically spaced bins:

        S(b) = sum_j  c_j * exp(-b * D_j)

    where D_j are the ``bins`` and c_j are the non-negative coefficients
    recovered by the NNLS solver.

    Args:
        d_range: (d_min, d_max) diffusion coefficient range.
        n_bins: Number of logarithmically spaced bins.
        **model_kwargs: Additional model configuration passed to the base class.

    Example:
        >>> import numpy as np
        >>> from pyneapple.models import NNLSModel
        >>>
        >>> model = NNLSModel(d_range=(1e-4, 0.1), n_bins=200)
        >>> bvalues = np.array([0, 50, 100, 200, 400, 800], dtype=float)
        >>> basis = model.get_basis(bvalues)  # shape (6, 200)
    """

    def __init__(
        self,
        d_range: tuple[float, float],
        n_bins: int,
        **model_kwargs,
    ) -> None:
        super().__init__(**model_kwargs)
        self.d_range = d_range
        self.n_bins = n_bins

    @property
    def bins(self) -> np.ndarray:
        """Return logarithmically spaced diffusion coefficient bins.

        Returns:
            Array of shape (n_bins,) with log-spaced D values.
        """
        return get_bins(self.d_range[0], self.d_range[1], self.n_bins)

    def get_basis(self, xdata: np.ndarray) -> np.ndarray:
        """Construct the exponential decay basis matrix.

        basis[i, j] = exp(-b[i] * D[j])

        Args:
            xdata: B-values, shape (n_measurements,).

        Returns:
            Basis matrix of shape (n_measurements, n_bins).

        Raises:
            ValueError: If xdata is not 1-D.
        """
        if xdata.ndim != 1:
            error_msg = (
                f"xdata must be a 1D array of shape (n_measurements,), "
                f"but got shape {xdata.shape}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return get_basis(xdata, self.bins)
