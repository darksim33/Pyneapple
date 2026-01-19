from pathlib import Path

from radimgarray import RadImgArray

from ..utils.logger import logger


def save_to_nifti(img: RadImgArray, filename: Path, **kwargs):
    """Save a RadImgArray to a NIfTI file.

    Args:
        img (RadImgArray): The image to save.
        filename (str): The filename to save the image to.
        **kwargs
            dtype (object): The data type to save the image as.
    """
    img.save(filename, save_as="nii", **kwargs)
    logger.info(f"Saved {filename}")


def arrays_to_nifti(imgs: list[RadImgArray], filenames: list[Path], **kwargs):
    """Save a list of RadImgArray to a NIfTI file.

    Args:
        imgs (list[RadImgArray]): The images to save.
        filename (str): The filename to save the images to.
        **kwargs
            dtype (object): The data type to save the images as.

    """
    for img, filename in zip(imgs, filenames):
        save_to_nifti(img, filename, **kwargs)
