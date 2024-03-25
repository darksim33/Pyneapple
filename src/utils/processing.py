import numpy as np
import warnings
from src.utils.utils import Nii, NiiSeg


def merge_nii_images(img1: Nii | NiiSeg, img2: Nii | NiiSeg) -> Nii:
    """
    Takes two Nii or NiiSeg objects and returns a new Nii object.

    The function first checks if the input images are of type NiiSeg, and if so, it compares their in-plane sizes.
    If they match, then the function multiplies each voxel value in img2 by its corresponding voxel value in img2.
    This is done for every slice of both images (i.e., for all time points). The resulting array is assigned to a
    new Nii object which is returned by the function.
    """

    array1 = img1.array.copy()
    array2 = img2.array.copy()
    if isinstance(img2, NiiSeg):
        if np.array_equal(array1.shape[0:2], array2.shape[0:2]):
            # compare in plane size of Arrays
            array_merged = np.ones(array1.shape)
            for idx in range(img1.array.shape[3]):
                array_merged[:, :, :, idx] = np.multiply(
                    array1[:, :, :, idx], array2[:, :, :, 0]
                )
            img_merged = img1.copy()
            img_merged.array = array_merged
            return img_merged
    else:
        warnings.warn("Warning: Secondary Image is not a mask!")


def get_mean_seg_signal(
    nii_img: Nii, nii_seg: NiiSeg, seg_index: int
) -> np.ndarray:
    img = nii_img.array.copy()
    seg_indexes = nii_seg.get_seg_indices(seg_index)
    number_of_b_values = img.shape[3]
    signal = np.zeros(number_of_b_values)
    for b_values in range(number_of_b_values):
        data = 0
        for idx in seg_indexes:
            idx[3] = b_values
            data = data + img[tuple(idx)]
        signal[b_values] = data / len(seg_indexes)
    return signal
