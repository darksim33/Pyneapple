"""B-value file I/O operations for DWI data.

This module provides functions for loading and validating b-value files
commonly used in diffusion-weighted imaging.
"""

import numpy as np
from pathlib import Path
from typing import List
from loguru import logger


def load_bvalues(path: str) -> np.ndarray:
    """Load b-values from a text file.

    Supports both line-separated and space-separated formats.
    Lines starting with '#' are treated as comments and ignored.
    Whitespace is automatically stripped.

    Args:
        path: Path to the b-value file (typically .txt or .bval).

    Returns:
        np.ndarray: 1D array of b-values (float64).

    Raises:
        FileNotFoundError: If the b-value file does not exist.
        ValueError: If the file is empty, contains non-numeric values, or
            has negative b-values.

    Examples:
        >>> bvalues = load_bvalues('data/bvalues.txt')
        >>> print(bvalues)
        [   0.   50.  100.  200.  400.  600.  800. 1000. 1500. 2000.]
    """
    path_obj = Path(path)

    # Check file exists
    if not path_obj.exists():
        logger.error(f"B-value file not found: {path}")
        raise FileNotFoundError(
            f"B-value file not found: {path}\n"
            f"Please check the file path and ensure the file exists."
        )

    # Read file
    try:
        with open(path_obj, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read b-value file {path}: {e}")
        raise ValueError(f"Failed to read b-value file: {path}\nError: {e}")

    # Parse lines
    bvalue_list: List[float] = []

    for line_num, line in enumerate(lines, start=1):
        # Strip whitespace
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Split by whitespace (handles both space-separated and single values per line)
        tokens = line.split()

        for token in tokens:
            try:
                value = float(token)
                bvalue_list.append(value)
            except ValueError:
                logger.error(f"Invalid b-value '{token}' on line {line_num} of {path}")
                raise ValueError(
                    f"Invalid b-value '{token}' on line {line_num} of file: {path}\n"
                    f"B-values must be numeric (integers or floats).\n"
                    f"Use '#' to mark comment lines."
                )

    # Check we got at least one b-value
    if len(bvalue_list) == 0:
        logger.error(f"No b-values found in file: {path}")
        raise ValueError(
            f"No b-values found in file: {path}\n"
            f"File is empty or contains only comments."
        )

    # Convert to numpy array
    bvalues = np.array(bvalue_list, dtype=np.float64)

    # Validate non-negative
    if np.any(bvalues < 0):
        negative_indices = np.where(bvalues < 0)[0]
        logger.error(f"Found {len(negative_indices)} negative b-values in {path}")
        raise ValueError(
            f"Found negative b-values in file: {path}\n"
            f"B-values must be non-negative (>= 0).\n"
            f"First few negative values: {bvalues[negative_indices[:5]]}"
        )

    logger.debug(f"Loaded {len(bvalues)} b-values from {path}: {bvalues}")

    return bvalues


def save_bvalues(bvalues: np.ndarray, path: str, format: str = "column") -> None:
    """Save b-values to a text file.

    Args:
        bvalues: 1D array of b-values.
        path: Output path for the b-value file.
        format: Format for output — ``'column'`` for one value per line,
            ``'row'`` for space-separated values (default: ``'column'``).

    Raises:
        ValueError: If bvalues is not 1D or contains negative values.

    Examples:
        >>> bvalues = np.array([0, 50, 100, 200, 400, 600, 800, 1000])
        >>> save_bvalues(bvalues, 'output/bvalues.txt', format='column')
    """
    # Validate input
    if bvalues.ndim != 1:
        logger.error(f"Expected 1D b-value array, got {bvalues.ndim}D")
        raise ValueError(
            f"Expected 1D b-value array, got {bvalues.ndim}D array.\n"
            f"Shape: {bvalues.shape}"
        )

    if np.any(bvalues < 0):
        logger.error("Cannot save negative b-values")
        raise ValueError(
            "B-values contain negative values.\n B-values must be non-negative (>= 0)."
        )

    # Create output directory if needed
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    try:
        with open(path_obj, "w") as f:
            f.write("# B-values for DWI acquisition\n")

            if format == "column":
                for bval in bvalues:
                    f.write(f"{bval:.1f}\n")
            elif format == "row":
                f.write(" ".join(f"{bval:.1f}" for bval in bvalues))
                f.write("\n")
            else:
                raise ValueError(f"Unknown format: {format}. Use 'column' or 'row'.")

        logger.info(f"Saved {len(bvalues)} b-values to: {path}")
    except Exception as e:
        logger.error(f"Failed to save b-value file {path}: {e}")
        raise ValueError(f"Failed to save b-value file: {path}\nError: {e}")
