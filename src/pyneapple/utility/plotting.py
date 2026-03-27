"""Plotting utilities for diffusion MRI fitting results.

This module provides functions to save parameter maps as heatmap images.
``matplotlib`` is an optional dependency; an :exc:`ImportError` is raised
with a helpful message when it is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger


def save_heatmap(
    param_map: np.ndarray,
    file_path: str | Path,
    *,
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    title: str | None = None,
) -> None:
    """Save a 2-D parameter map as a heatmap image.

    Args:
        param_map: 2-D array to visualise, shape ``(rows, cols)``.
        file_path: Output image path (e.g. ``results/D1_map.png``).
            Parent directories are created if needed.
        cmap: Matplotlib colormap name. Defaults to ``"hot"``.
        vmin: Lower bound for the colour scale. Defaults to array minimum.
        vmax: Upper bound for the colour scale. Defaults to array maximum.
        colorbar: Whether to include a colourbar. Defaults to ``True``.
        title: Optional title drawn above the axes.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
        ValueError: If ``param_map`` is not 2-D.

    Examples
    --------
    >>> save_heatmap(maps["D1"], "results/D1_map.png", vmin=0, vmax=0.01)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for heatmap export. "
            "Install it with: pip install matplotlib"
        ) from exc

    param_map = np.asarray(param_map)
    if param_map.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {param_map.shape}")

    fig, ax = plt.subplots()
    im = ax.imshow(param_map, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to: {file_path}")
