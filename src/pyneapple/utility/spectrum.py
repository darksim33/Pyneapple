"""NNLS spectrum post-processing utilities.

Functions for detecting peaks, applying cutoffs, and computing summary
statistics from NNLS diffusion spectra.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal


def calculate_peak_area(
    spectrum: np.ndarray,
    peak_indices: np.ndarray,
    peak_heights: np.ndarray,
    rel_height: float = 0.5,
) -> list[float]:
    """Gaussian area-under-curve correction for peaks in a regularized spectrum.

    Approximates each peak as a Gaussian and returns the integrated area using
    the FWHM of each peak.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D spectrum array.
    peak_indices : np.ndarray
        Indices of detected peaks in ``spectrum``.
    peak_heights : np.ndarray
        Heights (amplitudes) of the detected peaks.
    rel_height : float, optional
        Relative height at which to calculate the FWHM (default: 0.5 for standard FWHM).

    Returns
    -------
    list[float]
        Area-corrected fractions, one per peak.

    Examples
    --------
    >>> areas = calculate_peak_area(spectrum, peak_idx, peak_heights)
    """
    fwhms = scipy_signal.peak_widths(spectrum, peak_indices, rel_height=rel_height)[0]
    areas: list[float] = []
    for height, fwhm in zip(peak_heights, fwhms):
        area = float(height * fwhm / (2 * np.sqrt(2 * np.log(2))) * np.sqrt(2 * np.pi))
        areas.append(area)
    return areas


def find_spectrum_peaks(
    spectrum: np.ndarray,
    bins: np.ndarray,
    height: float = 0.1,
    regularized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Find peaks in a single NNLS spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D array of spectral coefficients, shape ``(n_bins,)``.
    bins : np.ndarray
        Diffusion coefficient bin values, shape ``(n_bins,)``.
    height : float, optional
        Minimum peak height threshold (default: 0.1).
    regularized : bool, optional
        If *True*, use Gaussian area-under-curve fractions instead of raw
        peak heights.  Should be set when the spectrum was produced with
        non-zero regularization order (default: False).

    Returns
    -------
    d_values : np.ndarray
        Diffusion coefficient values at detected peaks.
    f_values : np.ndarray
        Normalized fractions at detected peaks (sum to 1).

    Examples
    --------
    >>> d_vals, f_vals = find_spectrum_peaks(spectrum[i], model.bins,
    ...                                      regularized=True)
    """
    peak_indices, properties = scipy_signal.find_peaks(spectrum, height=height)

    if len(peak_indices) == 0:
        return np.array([]), np.array([])

    raw_heights = properties["peak_heights"]

    if regularized:
        f_values = np.array(calculate_peak_area(spectrum, peak_indices, raw_heights))
    else:
        f_values = raw_heights.copy()

    total = np.sum(f_values)
    if total > 0:
        f_values = f_values / total

    d_values = bins[peak_indices]
    return d_values, f_values


def geometric_mean_peak(
    positions: np.ndarray,
    heights: np.ndarray,
) -> tuple[float, float]:
    """Weighted geometric mean of multiple peaks in linear D-space.

    Used when merging peaks within a cutoff range.

    Parameters
    ----------
    positions : np.ndarray
        Peak positions in linear D-space (e.g. diffusion coefficients).
    heights : np.ndarray
        Peak weights / heights.

    Returns
    -------
    mean_position : float
        Weighted geometric mean position.
    total_height : float
        Sum of all heights.

    Examples
    --------
    >>> pos, height = geometric_mean_peak(d_vals, f_vals)
    """
    positions = np.asarray(positions, dtype=float)
    heights = np.asarray(heights, dtype=float)
    weighted_geomean = np.prod(positions ** (heights / np.sum(heights)))
    mean_position = float(np.log10(weighted_geomean))
    total_height = float(np.sum(heights))
    return mean_position, total_height


def apply_cutoffs(
    d_values: np.ndarray,
    f_values: np.ndarray,
    cutoffs: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge peaks within cutoff ranges into a single representative peak.

    For each cutoff range ``(lo, hi)``:

    - If no peaks fall in the range, ``nan`` is inserted for that component.
    - If exactly one peak falls in the range, it is kept as-is.
    - If multiple peaks fall in the range, they are merged using
      :func:`geometric_mean_peak`.

    Fractions are re-normalised after merging (ignoring ``nan`` entries).

    Parameters
    ----------
    d_values : np.ndarray
        Diffusion coefficient values at each peak.
    f_values : np.ndarray
        Fraction at each peak.
    cutoffs : list[tuple[float, float]]
        Ranges ``[(lo0, hi0), (lo1, hi1), …]`` that define expected
        diffusion components.

    Returns
    -------
    d_new : np.ndarray
        One D value per cutoff range (may contain ``nan``).
    f_new : np.ndarray
        Corresponding fractions, normalised to sum to 1 (ignoring ``nan``).

    Examples
    --------
    >>> d_new, f_new = apply_cutoffs(d_vals, f_vals,
    ...                              [(1e-4, 5e-3), (5e-3, 0.1)])
    """
    d_values = np.asarray(d_values, dtype=float)
    f_values = np.asarray(f_values, dtype=float)

    new_d: list[float] = []
    new_f: list[float] = []

    for lo, hi in cutoffs:
        mask = (d_values >= lo) & (d_values <= hi)
        _d = d_values[mask]
        _f = f_values[mask]

        if len(_d) == 0:
            new_d.append(float("nan"))
            new_f.append(float("nan"))
        elif len(_d) == 1:
            new_d.append(float(_d[0]))
            new_f.append(float(_f[0]))
        else:
            pos, height = geometric_mean_peak(_d, _f)
            new_d.append(pos)
            new_f.append(height)

    d_out = np.array(new_d)
    f_out = np.array(new_f)

    total = np.nansum(f_out)
    if total > 0:
        f_out = f_out / total

    return d_out, f_out
