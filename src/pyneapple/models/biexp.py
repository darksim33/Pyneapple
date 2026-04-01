"""Biexponential diffusion-weighted imaging (DWI) model.

This module implements the biexponential (IVIM) model used in diffusion MRI:

    Full:     S(b) = f1 * exp(-b * D1) + f2 * exp(-b * D2)
    Reduced:  S(b) = f1 * exp(-b * D1) + (1 - f1) * exp(-b * D2)
    S0:       S(b) = S0 * (f1 * exp(-b * D1) + (1 - f1) * exp(-b * D2))

with optional T1 relaxation correction:

    S_T1(b) = S(b) * (1 - exp(-TR / T1))                      (standard)
    S_STEAM(b) = S(b) * (1 - exp(-TR / T1)) * exp(-TM / T1)   (STEAM)

where:
- f1, f2 are volume fractions of each compartment
- D1, D2 are diffusion coefficients of each compartment
- S0 is the overall signal amplitude (S0 mode only)
- TR is the repetition time
- TM is the mixing time (STEAM sequence)
- T1 is the longitudinal relaxation time
"""

from __future__ import annotations

import numpy as np

from ..model_functions.multiexp import (
    apply_t1,
    apply_t1_jacobian,
    apply_t1_steam,
    biexp_forward,
    biexp_reduced_forward,
    biexp_s0_forward,
)
from .base import ParametricModel
from loguru import logger


class BiExpModel(ParametricModel):
    """Biexponential diffusion model for IVIM quantification.

    Supports three operating modes controlled by ``fit_reduced`` and ``fit_s0``:

    - **Reduced** (default): f2 = 1 - f1, params = [f1, D1, D2]
    - **Full**: f1, f2 independent, params = [f1, D1, f2, D2]
    - **S0**: Reduced with amplitude, params = [f1, D1, D2, S0]

    Optional T1 correction appends a T1 parameter.

    Args:
        fit_reduced: Constrain f2 = 1 - f1 (default True).
        fit_s0: Add S0 amplitude parameter (requires fit_reduced=True).
        fit_t1: Enable standard T1 relaxation fitting.
        fit_t1_steam: Enable STEAM T1 fitting (implies fit_t1=True).
        repetition_time: TR in ms (required when fit_t1=True).
        mixing_time: TM in ms (required when fit_t1_steam=True).
        **model_kwargs: Additional model configuration.

    Example:
        >>> import numpy as np
        >>> from pyneapple.models import BiExpModel
        >>>
        >>> model = BiExpModel()  # reduced by default
        >>> bvalues = np.array([0, 50, 100, 200, 400, 600, 800, 1000])
        >>> params = {'f1': 0.3, 'D1': 0.01, 'D2': 0.001}
        >>> signal = model.forward(bvalues, *params.values())
    """

    def __init__(
        self,
        fit_reduced: bool = True,
        fit_s0: bool = False,
        fit_t1: bool = False,
        fit_t1_steam: bool = False,
        repetition_time: float | None = None,
        mixing_time: float | None = None,
        **model_kwargs,
    ):
        super().__init__(**model_kwargs)
        if fit_s0 and not fit_reduced:
            raise ValueError(
                "fit_s0=True requires fit_reduced=True. "
                "Full model with independent fractions and S0 "
                "is over-parameterized."
            )
        # STEAM always implies standard T1 correction
        if fit_t1_steam:
            fit_t1 = True
        if fit_t1 and repetition_time is None:
            raise ValueError("repetition_time is required when fit_t1=True.")
        if fit_t1_steam and mixing_time is None:
            raise ValueError("mixing_time is required when fit_t1_steam=True.")

        self.fit_reduced = fit_reduced
        self.fit_s0 = fit_s0
        self.fit_t1 = fit_t1
        self.fit_t1_steam = fit_t1_steam
        self.repetition_time = repetition_time
        self.mixing_time = mixing_time
        self._validate_fixed_params()
        logger.debug("Initialized BiExpModel")

    @property
    def _all_param_names(self) -> list[str]:
        """Return ordered parameter names based on current mode.

        - Reduced: ['f1', 'D1', 'D2']
        - Full: ['f1', 'D1', 'f2', 'D2']
        - S0: ['f1', 'D1', 'D2', 'S0']
        - +T1: appends 'T1'
        """
        if self.fit_reduced:
            names = ["f1", "D1", "D2", "S0"] if self.fit_s0 else ["f1", "D1", "D2"]
        else:
            names = ["f1", "D1", "f2", "D2"]
        if self.fit_t1 or self.fit_t1_steam:
            names.append("T1")
        return names

    def forward(self, xdata: np.ndarray, *params: float) -> np.ndarray:
        """Forward model for optimization loop.

        Contains the canonical single-pixel physics implementation.

        Args:
            xdata: 1D array of b-values.
            params: Parameters as tuple or 1-D array.

                - Reduced (no S0): ``[f1, D1, D2]``
                - S0:              ``[f1, D1, D2, S0]``
                - Full:            ``[f1, D1, f2, D2]``
                - Any +T1:         appends ``T1``

        Returns:
            Signal array of shape ``(n_xdata,)``.
        """
        if self.fit_s0:
            f1, D1, D2, S0 = params[0], params[1], params[2], params[3]
            signal = biexp_s0_forward(xdata, f1, D1, D2, S0)
        elif self.fit_reduced:
            f1, D1, D2 = params[0], params[1], params[2]
            signal = biexp_reduced_forward(xdata, f1, D1, D2)
        else:
            f1, D1, f2, D2 = params[0], params[1], params[2], params[3]
            signal = biexp_forward(xdata, f1, D1, f2, D2)

        if self.fit_t1:
            t1_index = self._all_param_names.index("T1")
            T1 = params[t1_index]
            if self.repetition_time is not None:
                signal = apply_t1(signal, self.repetition_time, T1)
            else:
                raise ValueError("repetition_time is required for T1 correction.")
            if self.fit_t1_steam and self.mixing_time is not None:
                signal = apply_t1_steam(signal, self.mixing_time, T1)

        return signal

    def jacobian(self, xdata: np.ndarray, *params: float) -> np.ndarray | None:
        """Jacobian for optimization loop.

        Contains the canonical single-pixel Jacobian implementation. Column order matches ``_all_param_names``.

        Args:
            xdata: 1D array of b-values.
            params: Parameters as tuple or 1-D array (same order as
                ``forward``).

        Returns:
            ``(n_xdata, n_params)`` Jacobian array, or ``None``.
        """
        f1, D1 = params[0], params[1]
        exp1 = np.exp(-xdata * D1)

        if self.fit_s0:
            D2, S0 = params[2], params[3]
            exp2 = np.exp(-xdata * D2)
            cols = [
                S0 * (exp1 - exp2),  # dS/df1
                -xdata * S0 * f1 * exp1,  # dS/dD1
                -xdata * S0 * (1 - f1) * exp2,  # dS/dD2
                f1 * exp1 + (1 - f1) * exp2,  # dS/dS0
            ]
            base_signal = S0 * (f1 * exp1 + (1 - f1) * exp2)
        elif self.fit_reduced:
            D2 = params[2]
            exp2 = np.exp(-xdata * D2)
            cols = [
                exp1 - exp2,  # dS/df1
                -xdata * f1 * exp1,  # dS/dD1
                -xdata * (1 - f1) * exp2,  # dS/dD2
            ]
            base_signal = f1 * exp1 + (1 - f1) * exp2
        else:
            # Full mode: params = [f1, D1, f2, D2]
            f2, D2 = params[2], params[3]
            exp2 = np.exp(-xdata * D2)
            cols = [
                exp1,  # dS/df1
                -xdata * f1 * exp1,  # dS/dD1
                exp2,  # dS/df2
                -xdata * f2 * exp2,  # dS/dD2
            ]
            base_signal = f1 * exp1 + f2 * exp2

        jac = np.column_stack(cols)
        if not self.fit_t1:
            return jac

        # T1: scale base derivatives and append dS/dT1
        T1 = params[self._all_param_names.index("T1")]
        if self.repetition_time is None:
            raise ValueError("repetition_time is required for T1 Jacobian.")
        else:
            return apply_t1_jacobian(
                jac=jac,
                base_signal=base_signal,
                T1=T1,
                repetition_time=self.repetition_time,
                mixing_time=self.mixing_time if self.fit_t1_steam else None,
            )
