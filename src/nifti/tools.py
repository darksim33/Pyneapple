from __future__ import annotations

import numpy as np
from . import NiiSeg


def get_single_seg_array(seg: NiiSeg, seg_number: int | str) -> np.ndarray:
    """
    Get array only containing one segmentation set to one.

    Parameters
    ----------
    seg_number: int | str
        Number of segmentation to process.

    Returns
    -------
    array: np.ndarray
        array containing an array showing only the segmentation of the selected segment number.

    """
    array = np.zeros(seg.array.shape)
    indices = seg.get_seg_indices(seg_number)
    for idx, value in zip(indices, np.ones(len(indices))):
        try:
            array[idx] = value
        except ValueError:
            raise ValueError(f"Index {idx} out of array shape {array.shape}")
    return array


def get_mean_signal(img: np.ndarray, seg: NiiSeg, seg_number: int):
    """
    Get mean signal of Pixel included in segmentation.

    Parameters
    ----------
    img: np.ndarray
        image to process
    seg: NiiSeg
        Segmentation to process
    seg_number: int | np.integer
        Number of segmentation to process

    Returns
    -------
    mean_signal: np.ndarray
        Mean signal of the selected Pixels for given Segmentation
    """
    signals = list(
        img[i, j, k, :]
        for i, j, k in zip(
            *np.nonzero(np.squeeze(seg.get_single_seg_array(seg_number), axis=3))
        )
    )
    return np.mean(signals, axis=0)
