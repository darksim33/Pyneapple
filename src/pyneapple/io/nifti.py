from pathlib import Path

from radimgarray import RadImgArray

from ..utils.logger import logger


def save_to_nifti(img: RadImgArray, filename: Path, **kwargs):
    """Save a RadImgArray to a NIfTI file.

    Parameters
    ----------
    img : RadImgArray
        The image to save.
    filename : str
        The filename to save the image to.
    **kwargs
        Additional keyword arguments to pass to the save method.
        dtype : object
            The data type to save the image as.

    Returns
    -------
    None
    """
    img.save(filename, save_as="nii", **kwargs)
    logger.info(f"Saved {filename}")


def arrays_to_nifti(imgs: list[RadImgArray], filenames: list[Path], **kwargs):
    """Save a list of RadImgArray to a NIfTI file.

    Parameters
    ----------
    imgs : list[RadImgArray]
        The images to save.
    filename : str
        The filename to save the images to.
    **kwargs
        Additional keyword arguments to pass to the save method.
        dtype : object
            The data type to save the images as.

    Returns
    -------
    None
    """
    for img, filename in zip(imgs, filenames):
        save_to_nifti(img, filename, **kwargs)
