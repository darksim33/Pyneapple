from __future__ import annotations

import numpy as np

from ..utils.logger import logger
from radimgarray import RadImgArray


class ResultDict(dict):
    """Custom dictionary for storing fitting results and returning them according to
    fit style.

    Basic dictionary enhanced with fit type utility to store results of pixel or
    segmentation wise fitting. Some methods are overwritten to handle either of the
    two fit types.

    Attributes:
        fit_type (str): Type of fit process. Either "Pixel" or "Segmentation".
        identifier (dict): Dictionary containing pixel -> segmentation value pairs.
            {(0,0,0): 1, (1,0,0): 2...}

    Methods:
        set_segmentation_wise(self, identifier: dict) Update the dictionary for
            segmented fitting
        as_array(self, shape: dict) -> np.ndarray Return array containing the dict
            content
    """

    def __init__(self, fit_type: str | None = None, identifier: dict | None = None):
        """Initialize CustomDict object.

        Args:
            fit_type (str): Type of fit process. Either "Pixel" or "Segmentation".
            identifier (dict): Dictionary containing pixel to segmentation value pairs.
        """
        super().__init__()
        self.type = fit_type
        self.identifier = identifier
        if fit_type == "Segmentation" and identifier is None:
            error_msg = "Identifier is required if fit_type is 'Segmentation'"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def __getitem__(self, key):
        """Return value of key in dictionary.

        The __getitem__ method is overwritten to handle either pixel or segmentation
        based keys.

        Args:
            key: Key to look up in dictionary.
        Returns:
            value: Value of key in dictionary.
        """
        value = []
        key = self.validate_key(key)
        if isinstance(key, tuple):
            # If the key is a tuple containing the pixel coordinates:
            try:
                if self.type == "Segmentation":
                    # in case of Segmentation wise fitting the identifier
                    # dict is needed to look up pixel segmentation number
                    value = super().__getitem__(self.identifier[key])
                else:
                    value = super().__getitem__(key)
            except KeyError:
                error_msg = f"Key '{key}' not found in dictionary."
                logger.error(error_msg)
                raise KeyError(error_msg)
        elif isinstance(key, int):
            # If the key is an int for the segmentation:
            try:
                if self.type == "Segmentation":
                    value = super().__getitem__(key)
            except KeyError:
                error_msg = f"Key '{key}' not found in dictionary."
                logger.error(error_msg)
                raise KeyError(error_msg)
        return value

    def __setitem__(self, key, value):
        """Set value of key in dictionary."""
        key = self.validate_key(key)
        super().__setitem__(key, value)

    def get(self, key, default=None):
        """Return value of key in dictionary."""
        try:
            return self.__getitem__(key)
        except KeyError:
            if default is not None:
                return default
            else:
                error_msg = f"Key '{key}' not found in dictionary."
                logger.error(error_msg)
                raise KeyError(error_msg)

    @staticmethod
    def validate_key(key):
        """Validate key type.
        Check weather the given key is supported by the CustomDict."""
        if isinstance(key, tuple):
            pass
        elif isinstance(key, int):
            pass
        elif isinstance(key, str):
            error_msg = "String assignment and calling is not supported."
            logger.error(error_msg)
            raise TypeError(error_msg)
        elif isinstance(key, float):
            error_msg = "Float assignment and calling is not supported."
            logger.error(error_msg)
            raise TypeError(error_msg)
        try:
            if np.issubdtype(key, np.integer):
                key = int(key)
        except TypeError:
            pass
        return key

    def set_segmentation_wise(self, identifier: dict | None = None):
        """
        Update segmentation info of dict.

        Args:
            identifier (dict): Dictionary containing pixel to segmentation value pairs.
        """
        if isinstance(identifier, dict):
            self.identifier = identifier  # .copy()
            self.type = "Segmentation"
        elif identifier is None or False:
            self.identifier = {}
            self.type = "Pixel"

    def as_array(self, shape: tuple | list, **kwargs) -> np.ndarray:
        """Returns a numpy array of the dict fit data.

        Args:
            shape (tuple): Shape of final fit data (minimum 4D).

        Returns:
            array (np.ndarray): Numpy array of the dict fit data.
        """
        if isinstance(shape, tuple):
            shape = list(shape)

        if len(shape) < 4:
            error_msg = "Shape must be at least 4 dimensions."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            if isinstance(list(self.values())[0], (np.ndarray, list)):
                shape[3] = list(self.values())[0].shape[
                    0
                ]  # read shape of first array in dict to determine shape
            else:  # if there is only a single value
                shape[3] = 1
        array = np.zeros(shape)

        if self.type == "Segmentation":
            for key, seg_number in self.identifier.items():
                array[key] = self[seg_number]
        else:
            for key, value in self.items():
                array[key] = value
        return array

    def as_RadImgArray(self, img: RadImgArray, **kwargs) -> RadImgArray:
        """Returns a RadImgArray of the dict fit data."""
        array = self.as_array(img.shape, **kwargs)
        return RadImgArray(array, img.info)
