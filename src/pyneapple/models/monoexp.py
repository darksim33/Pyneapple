"""Monoexponential diffusion-weighted imaging (DWI) model.

This module implements the standard monoexponential ADC (Apparent Diffusion
Coefficient) model used in diffusion MRI:

    S(b) = S0 * exp(-b * D)

with optional T1 relaxation correction:

    S_T1(b) = S(b) * (1 - exp(-TR / T1))                      (standard)
    S_STEAM(b) = S(b) * (1 - exp(-TR / T1)) * exp(-TM / T1)   (STEAM)

where:
- S(b) is the measured signal at b-value b
- S0 is the signal intensity at b=0 (no diffusion weighting)
- D is the apparent diffusion coefficient (ADC)
- b is the diffusion weighting factor (b-value)
- TR is the repetition time
- TM is the mixing time (STEAM sequence)
- T1 is the longitudinal relaxation time
"""

import numpy as np

from ..model_functions.multiexp import (
    apply_t1,
    apply_t1_jacobian,
    apply_t1_steam,
    monoexp_forward,
)

from loguru import logger
from .base import ParametricModel


class MonoExpModel(ParametricModel):
    """Monoexponential DWI model for ADC quantification.

    The forward model is:
        S(b) = S0 * exp(-b * D)

    With optional T1 correction (fit_t1=True):
        S(b) = S0 * exp(-b * D) * (1 - exp(-TR / T1))

    With STEAM T1 correction (fit_t1_steam=True, implies fit_t1=True):
        S(b) = S0 * exp(-b * D) * (1 - exp(-TR / T1)) * exp(-TM / T1)

    Args:
        fit_t1: Enable T1 relaxation fitting (standard).
        fit_t1_steam: Enable STEAM T1 fitting (implies fit_t1=True).
        repetition_time: TR in ms (required if fit_t1=True).
        mixing_time: TM in ms (required if fit_t1_steam=True).
        **model_kwargs: Additional model configuration.

    Example:
        >>> import numpy as np
        >>> from salak.models import MonoExpModel
        >>>
        >>> model = MonoExpModel()
        >>> bvalues = np.array([0, 50, 100, 200, 400, 600, 800, 1000])
        >>> params = {'S0': 1000.0, 'D': 0.001}
        >>> signal = model.forward(bvalues, *params.values())
    """

    def __init__(
        self,
        fit_t1: bool = False,
        fit_t1_steam: bool = False,
        repetition_time: float | None = None,
        mixing_time: float | None = None,
        **model_kwargs,
    ):
        super().__init__(**model_kwargs)
        # STEAM always implies standard T1 correction
        if fit_t1_steam:
            fit_t1 = True
        if fit_t1 and repetition_time is None:
            raise ValueError("repetition_time is required when fit_t1=True.")
        if fit_t1_steam and mixing_time is None:
            raise ValueError("mixing_time is required when fit_t1_steam=True.")

        self.fit_t1 = fit_t1
        self.fit_t1_steam = fit_t1_steam
        self.repetition_time = repetition_time
        self.mixing_time = mixing_time
        self._validate_fixed_params()
        logger.debug("Initialized MonoExpModel")

    @property
    def _all_param_names(self) -> list[str]:
        """Return ordered list of parameter names.

        Returns ['S0', 'D'] by default, or ['S0', 'D', 'T1'] when T1 fitting
        is enabled.
        """
        names = ["S0", "D"]
        if self.fit_t1 or self.fit_t1_steam:
            names.append("T1")
        return names

    def forward(self, xdata: np.ndarray, *params: float) -> np.ndarray:
        """Forward model for optimization loop.

        Contains the canonical single-pixel physics implementation.

        Args:
            xdata: 1D array of b-values.
            params: Parameters as tuple or 1-D array.
                - Non-T1: ``[S0, D]``
                - T1/STEAM: ``[S0, D, T1]``

        Returns:
            Signal array of shape ``(n_xdata,)``.
        """
        S0, D = params[0], params[1]
        signal = monoexp_forward(xdata, S0, D)

        if self.fit_t1 and self.repetition_time is not None:
            t1_index = self._all_param_names.index("T1")
            T1 = params[t1_index]
            signal = apply_t1(signal, self.repetition_time, T1)
            if self.fit_t1_steam and self.mixing_time is not None:
                signal = apply_t1_steam(signal, self.mixing_time, T1)

        return signal

    def jacobian(self, xdata: np.ndarray, *params: float) -> np.ndarray | None:
        """Jacobian for optimization loop.

        Contains the canonical single-pixel Jacobian implementation.

        Args:
            xdata: 1D array of b-values.
            params: Parameters as tuple or 1-D array.
                - Non-T1: ``[S0, D]``
                - T1/STEAM: ``[S0, D, T1]``

        Returns:
            ``(n_xdata, n_params)`` Jacobian array, or ``None``.
        """
        S0, D = params[0], params[1]
        exp_term = np.exp(-xdata * D)
        jac_S0 = exp_term  # dS/dS0 = exp(-b*D)
        jac_D = -xdata * S0 * exp_term  # dS/dD  = -b * S0 * exp(-b*D)

        jac = np.column_stack([jac_S0, jac_D])
        if not self.fit_t1:
            return jac

        # T1: scale base derivatives and append dS/dT1
        T1 = params[self._all_param_names.index("T1")]
        if self.repetition_time is None:
            raise ValueError("repetition_time is required for T1 Jacobian.")
        else:
            return apply_t1_jacobian(
                jac=jac,
                base_signal=S0 * exp_term,
                T1=T1,
                repetition_time=self.repetition_time,
                mixing_time=self.mixing_time if self.fit_t1_steam else None,
            )
