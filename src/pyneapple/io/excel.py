"""Excel export utilities for diffusion MRI fitting results.

This module provides functions to export fitted parameters and NNLS spectra to
Excel files.  ``pandas`` is an optional dependency; an :exc:`ImportError` is
raised with a helpful message when it is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger


def save_params_to_excel(
    fitted_params: dict[str, np.ndarray],
    pixel_indices: list[tuple[int, ...]],
    file_path: str | Path,
) -> None:
    """Save fitted parameters to an Excel file (one row per pixel).

    Each parameter becomes a set of columns; multi-valued parameters (e.g.
    shape ``(n_pixels, k)``) produce ``k`` columns named ``<param>_0``,
    ``<param>_1``, …

    Parameters
    ----------
    fitted_params : dict[str, ndarray]
        Dictionary of parameter name → 1-D or 2-D array.  The first
        dimension must equal ``len(pixel_indices)``.
    pixel_indices : list[tuple[int, ...]]
        Spatial index ``(x, y, z)`` for each row.
    file_path : str or Path
        Output ``.xlsx`` path.  Parent directories are created if needed.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.
    ValueError
        If ``fitted_params`` is empty or array lengths are inconsistent.

    Examples
    --------
    >>> save_params_to_excel(fitter.fitted_params_, fitter.pixel_indices,
    ...                      "results/params.xlsx")
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for Excel export. "
            "Install it with: pip install pandas openpyxl"
        ) from exc

    if not fitted_params:
        raise ValueError("fitted_params is empty — nothing to export.")

    rows: list[dict] = []
    for i, idx in enumerate(pixel_indices):
        row: dict = {"x": idx[0], "y": idx[1]}
        if len(idx) >= 3:
            row["z"] = idx[2]

        for param, values in fitted_params.items():
            v = values[i]
            if np.ndim(v) == 0:
                row[param] = float(v)
            else:
                v = np.atleast_1d(v)
                for j, val in enumerate(v):
                    row[f"{param}_{j}"] = float(val)

        rows.append(row)

    df = pd.DataFrame(rows)
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(file_path, index=False)
    logger.info(f"Saved parameter table to: {file_path}")


def save_spectrum_to_excel(
    spectrum: np.ndarray,
    pixel_indices: list[tuple[int, ...]],
    bins: np.ndarray,
    file_path: str | Path,
) -> None:
    """Save an NNLS spectrum to an Excel file (one row per pixel).

    Columns are the log-spaced diffusion coefficient bins; index columns
    hold the spatial coordinates.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum array of shape ``(n_pixels, n_bins)``.
    pixel_indices : list[tuple[int, ...]]
        Spatial index for each row of ``spectrum``.
    bins : np.ndarray
        Diffusion coefficient bin values, shape ``(n_bins,)``.
    file_path : str or Path
        Output ``.xlsx`` path.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.
    ValueError
        If ``spectrum`` is not 2-D or shapes are inconsistent.

    Examples
    --------
    >>> save_spectrum_to_excel(spectrum, fitter.pixel_indices, model.bins,
    ...                        "results/spectrum.xlsx")
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for Excel export. "
            "Install it with: pip install pandas openpyxl"
        ) from exc

    spectrum = np.asarray(spectrum)
    if spectrum.ndim != 2:
        raise ValueError(
            f"Expected 2-D spectrum array (n_pixels, n_bins), got shape {spectrum.shape}"
        )
    if spectrum.shape[1] != len(bins):
        raise ValueError(
            f"spectrum has {spectrum.shape[1]} bins but bins array has {len(bins)} entries."
        )

    index_cols: list[dict] = []
    for idx in pixel_indices:
        entry: dict = {"x": idx[0], "y": idx[1]}
        if len(idx) >= 3:
            entry["z"] = idx[2]
        index_cols.append(entry)

    index_df = pd.DataFrame(index_cols)
    bin_cols = pd.DataFrame(spectrum, columns=[f"{b:.6g}" for b in bins])
    df = pd.concat([index_df, bin_cols], axis=1)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(file_path, index=False)
    logger.info(f"Saved spectrum table to: {file_path}")
