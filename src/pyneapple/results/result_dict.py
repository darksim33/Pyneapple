from __future__ import annotations

from copy import deepcopy

import numpy as np

from radimgarray import RadImgArray

from ..utils.logger import logger


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

    def __init__(self, fit_type: str | None = None, identifier: dict = {}):
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
        value = None
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

    def deepcopy(self):
        """Return a deep copy of the dictionary."""
        new_dict = ResultDict(self.type, self.identifier)
        new_dict.update(deepcopy(self))
        return new_dict

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
            if isinstance(key, np.integer):
                key = int(key)
        except TypeError:
            error_msg = f"Unsupported key type {type(key)}."
            logger.error(error_msg)
            raise TypeError(error_msg)
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
                # Get maximum length across all values
                max_len = (
                    max(self._get_length(v) for v in self.values())
                    if self.values()
                    else 1
                )
                shape[3] = max_len
            else:  # if there is only a single value
                shape[3] = 1
        array = np.zeros(shape)

        if self.type == "Segmentation":
            for key, seg_number in self.identifier.items():
                array[key] = self[seg_number]
        else:
            for key, value in self.items():
                if self._get_length(value) < shape[-1]:
                    _value = np.zeros(shape[-1])
                    _value[: len(value)] = value
                elif self._get_length(value) > shape[-1]:
                    error_msg = "Error: Value shape is larger than target array shape!"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    _value = value
                array[key] = _value
        return array

    def _get_length(self, value):
        # Helper function to get length of value
        if isinstance(value, list):
            return len(value)
        elif isinstance(value, np.ndarray):
            if not value.size == 1:
                return len(value)
            else:
                return 1
        else:
            return 1

    def as_RadImgArray(self, img: RadImgArray, **kwargs) -> RadImgArray:
        """Returns a RadImgArray of the dict fit data."""
        array = self.as_array(img.shape, **kwargs)
        return RadImgArray(array, img.info)
