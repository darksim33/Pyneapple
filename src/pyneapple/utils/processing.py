from __future__ import annotations
import numpy as np
import warnings
from radimgarray import RadImgArray, SegImgArray


def merge_nii_images(
    img1: RadImgArray | SegImgArray, img2: RadImgArray | SegImgArray
) -> SegImgArray:
    """Takes two Nii or NiiSeg objects and returns a new Nii object.

    The function first checks if the input images are of type SegImgArray, and if so, it
    compares their in-plane sizes. If they match, then the function multiplies each
    voxel value in img2 by its corresponding voxel value in img2. This is done for every
    slice of both images (i.e., for all time points). The resulting array is assigned to
    a new RadImgArray object which is returned by the function.

    Args:
        img1 ( RadImgArray, SegImgArray): A RadImgArray or SegImgArray object.
        img2 ( RadImgArray, SegImgArray): A RadImgArray or SegImgArray object.
    Returns:
        RadImgArray: A new RadImgArray object.
    """
    array1 = img1.copy()
    array2 = img2.copy()
    if isinstance(img2, SegImgArray):
        if np.array_equal(array1.shape[0:2], array2.shape[0:2]):
            # compare in plane size of Arrays
            array_merged = np.ones(array1.shape)
            for idx in range(img1.shape[3]):
                array_merged[:, :, :, idx] = np.multiply(
                    array1[:, :, :, idx], array2[:, :, :, 0]
                )
            img_merged = img1.copy()
            img_merged.array = array_merged
            return img_merged
    else:
        warnings.warn("Warning: Secondary Image is not a mask!")


def get_mean_seg_signal(
    img: RadImgArray, seg: SegImgArray, seg_index: int
) -> np.ndarray:
    """Takes a Nii and NiiSeg object and returns the mean signal of a segmented region.

    Args:
        img (RadImgArray): A RadImgArray object.
        seg (SegImgArray): A SegImgArray object.
        seg_index (int): The index of the segmented region.
    Returns:
        np.ndarray: The mean signal of the segmented region.
    """
    _img = img.copy()
    seg_indexes = seg.get_seg_indices(seg_index)
    number_of_b_values = _img.shape[3]
    signal = np.zeros(number_of_b_values)
    for b_values in range(number_of_b_values):
        data = 0
        for idx in seg_indexes:
            idx[3] = b_values
            data = data + _img[tuple(idx)]
        signal[b_values] = data / len(seg_indexes)
    return signal
