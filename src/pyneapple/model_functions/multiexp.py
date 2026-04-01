"""Pure forward model functions for multi-exponential diffusion models.

Stateless functions implementing the signal equations for mono-, bi-, and
tri-exponential diffusion models. These are the mathematical core that can
be used by both legacy model classes and new ParametricModel classes.

All functions follow the signature:
    f(xdata, *params) -> signal

where xdata and signal are numpy arrays.

Functions:
    Monoexponential:
        monoexp_forward, monoexp_reduced_forward

    Biexponential:
        biexp_forward, biexp_reduced_forward, biexp_s0_forward

    Triexponential:
        triexp_forward, triexp_reduced_forward, triexp_s0_forward

    T1 modifiers:
        apply_t1, apply_t1_steam
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# Monoexponential
# =============================================================================


def monoexp_forward(xdata: np.ndarray, S0: float, D: float) -> np.ndarray:
    """Monoexponential model: S(b) = S0 * exp(-b * D).

    Args:
        xdata: B-values, shape (n,).
        S0: Signal at b=0.
        D: Apparent diffusion coefficient.

    Returns:
        Signal array, shape (n,).
    """
    return S0 * np.exp(-xdata * D)


def monoexp_reduced_forward(xdata: np.ndarray, D: float) -> np.ndarray:
    """Reduced monoexponential: S(b) = exp(-b * D).

    Args:
        xdata: B-values, shape (n,).
        D: Apparent diffusion coefficient.

    Returns:
        Signal array, shape (n,).
    """
    return np.exp(-xdata * D)


# =============================================================================
# Biexponential
# =============================================================================


def biexp_forward(
    xdata: np.ndarray, f1: float, D1: float, f2: float, D2: float
) -> np.ndarray:
    """Full biexponential: S = f1 * exp(-D1 * b) + f2 * exp(-D2 * b).

    Args:
        xdata: B-values, shape (n,).
        f1: Volume fraction of component 1.
        D1: Diffusion coefficient of component 1.
        f2: Volume fraction of component 2.
        D2: Diffusion coefficient of component 2.

    Returns:
        Signal array, shape (n,).
    """
    return f1 * np.exp(-xdata * D1) + f2 * np.exp(-xdata * D2)


def biexp_reduced_forward(
    xdata: np.ndarray, f1: float, D1: float, D2: float
) -> np.ndarray:
    """Reduced biexponential: S = f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b).

    The second fraction is constrained as :math:`f_2 = 1 - f_1`.

    Args:
        xdata: B-values, shape (n,).
        f1: Volume fraction of component 1.
        D1: Diffusion coefficient of component 1.
        D2: Diffusion coefficient of component 2.

    Returns:
        Signal array, shape (n,).
    """
    return f1 * np.exp(-xdata * D1) + (1 - f1) * np.exp(-xdata * D2)


def biexp_s0_forward(
    xdata: np.ndarray, f1: float, D1: float, D2: float, S0: float
) -> np.ndarray:
    """Biexponential with S0: S = S0 * (f1 * exp(-D1 * b) + (1 - f1) * exp(-D2 * b)).

    Args:
        xdata: B-values, shape (n,).
        f1: Volume fraction of component 1.
        D1: Diffusion coefficient of component 1.
        D2: Diffusion coefficient of component 2.
        S0: Signal intensity at b=0.

    Returns:
        Signal array, shape (n,).
    """
    return S0 * (f1 * np.exp(-xdata * D1) + (1 - f1) * np.exp(-xdata * D2))


# =============================================================================
# Triexponential
# =============================================================================


def triexp_forward(
    xdata: np.ndarray,
    f1: float,
    D1: float,
    f2: float,
    D2: float,
    f3: float,
    D3: float,
) -> np.ndarray:
    """Full triexponential: S = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + f3 * exp(-D3 * b).

    Args:
        xdata: B-values, shape (n,).
        f1, f2, f3: Volume fractions
        D1, D2, D3: Diffusion coefficients

    Returns:
        Signal array, shape (n,).
    """
    return (
        f1 * np.exp(-xdata * D1) + f2 * np.exp(-xdata * D2) + f3 * np.exp(-xdata * D3)
    )


def triexp_reduced_forward(
    xdata: np.ndarray,
    f1: float,
    D1: float,
    f2: float,
    D2: float,
    D3: float,
) -> np.ndarray:
    """Reduced triexponential: S = f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b).

    The third fraction is constrained as :math:`f_3 = 1 - f_1 - f_2`.

    Args:
        xdata: B-values, shape (n,).
        f1, f2: Volume fractions (f3 = 1 - f1 - f2)
        D1, D2, D3: Diffusion coefficients

    Returns:
        Signal array, shape (n,).
    """
    return (
        f1 * np.exp(-xdata * D1)
        + f2 * np.exp(-xdata * D2)
        + (1 - f1 - f2) * np.exp(-xdata * D3)
    )


def triexp_s0_forward(
    xdata: np.ndarray,
    f1: float,
    D1: float,
    f2: float,
    D2: float,
    D3: float,
    S0: float,
) -> np.ndarray:
    """Triexponential with S0: S = S0 * (f1 * exp(-D1 * b) + f2 * exp(-D2 * b) + (1 - f1 - f2) * exp(-D3 * b)).

    Args:
        xdata: B-values, shape (n,).
        f1, f2: Volume fractions (f3 = 1 - f1 - f2)
        D1, D2, D3: Diffusion coefficients
        S0: Signal intensity at b=0

    Returns:
        Signal array, shape (n,).
    """
    return S0 * (
        f1 * np.exp(-xdata * D1)
        + f2 * np.exp(-xdata * D2)
        + (1 - f1 - f2) * np.exp(-xdata * D3)
    )


# =============================================================================
# T1 modifiers
# =============================================================================


def apply_t1(signal: np.ndarray, repetition_time: float, T1: float) -> np.ndarray:
    """Apply T1 relaxation correction: S * (1 - exp(-TR / T1)).

    Args:
        signal: Input signal to modify.
        repetition_time: TR in same units as T1.
        T1: T1 relaxation time.

    Returns:
        T1-corrected signal.
    """
    return signal * (1 - np.exp(-repetition_time / T1))


def apply_t1_steam(signal: np.ndarray, mixing_time: float, T1: float) -> np.ndarray:
    """Apply STEAM T1 relaxation correction: S * exp(-TM / T1).

    This is always used **in addition to** :func:`apply_t1`, not as a
    replacement.  The full STEAM-corrected signal is:

    STEAM
        S(T1) = S * (1 - exp(-TR / T1)) * exp(-TM / T1)

    Args:
        signal: Input signal (typically already corrected by :func:`apply_t1`)
        mixing_time: TM (mixing time) in same units as T1
        T1: T1 relaxation time

    Returns:
        STEAM T1-corrected signal
    """
    return signal * np.exp(-mixing_time / T1)


def apply_t1_jacobian(
    jac: np.ndarray,
    base_signal: np.ndarray,
    T1: float | np.ndarray,
    repetition_time: float,
    mixing_time: float | None = None,
) -> np.ndarray:
    """Apply T1 correction to an existing Jacobian array.

    Scales all entries in *jac* by the T1 correction factor and appends
    a ``T1`` derivative column.

    **T1 only** (``mixing_time is None``):

        factor = 1 - exp(-TR / T1)

        dS/dT1 = -S_base * exp(-TR / T1) * (TR / T1^2)

    **STEAM** (``mixing_time`` provided):

        factor = (1 - exp(-TR / T1)) * exp(-TM / T1)

        dS/dT1 = S_base * exp(-TM / T1) / T1^2 * (-TR * exp(-TR / T1) + TM * (1 - exp(-TR / T1)))

    Args:
        jac: Jacobian array of shape (n_bvalues, n_params).
        base_signal: Pre-T1-correction signal, shape (n_bvalues,).
        T1: Longitudinal relaxation time (scalar or array, broadcasted if needed).
        repetition_time: TR in same units as *T1*.
        mixing_time: TM in same units as *T1*. ``None`` for standard T1 correction only.

    Returns:
        Updated Jacobian array with an additional column for ``T1`` derivatives.
    """
    TR = repetition_time
    exp_TR = np.exp(-TR / T1)
    A = 1 - exp_TR  # standard T1 factor

    if mixing_time is not None:
        TM = mixing_time
        exp_TM = np.exp(-TM / T1)
        t1_factor = A * exp_TM
        jac_T1 = base_signal * exp_TM / T1**2 * (-TR * exp_TR + TM * A)
    else:
        t1_factor = A
        jac_T1 = base_signal * (-exp_TR * TR / T1**2)

    # Scale all existing Jacobian columns by the T1 correction factor.
    # t1_factor may be a scalar or a 1-D array (spatial fitting); both cases
    # are handled explicitly so that scalar indexing works correctly.
    if np.ndim(t1_factor) > 0:
        jac = jac * t1_factor[:, np.newaxis]
    else:
        jac = jac * t1_factor

    # Append the new T1 derivative column
    jac = np.column_stack((jac, jac_T1))

    return jac
