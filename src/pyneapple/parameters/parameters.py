"""Module for Parameters base class and subclasses.
The Parameter classes are the heart of the fitting process in Pyneapple. They contain
all the necessary information for the fitting process and are used to streamline the
fitting process for all kinds of models. The Parameters class is an abstract base class
that defines the basic structure for all fitting parameters subclasses. The subclasses
contain the specific information for the fitting process, such as the fitting model,
fitting function, and scaling of the image data. The Parameters class is used to
initialize the fitting parameters and load them from a .json file. The subclasses are
used to define the specific fitting parameters for the different fitting models. The
Parameters class also contains methods to save and load the fitting parameters from a
.json file and to get the pixel and segment arguments for the fitting process.

Classes:
    Params: Abstract base class for Parameters child class.
    Parameters: Base class for all fitting parameters subclasses.
"""

from __future__ import annotations
from collections.abc import Callable

import numpy as np
import json

from functools import partial
from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty

# from nifti import NiiSeg, tools
from radimgarray import RadImgArray, SegImgArray, tools
from ..utils.exceptions import ClassMismatch

# from ..results import CustomDict
from ..parameters import Boundaries


class Params(ABC):
    """Abstract base class for Parameters child class.

    Defines abstract properties and methods for Parameters child classes.
    """

    def __init__(self):
        self._scale_image: str | int = ""
        self._fit_model = lambda: None
        self._fit_function = lambda: None

    @abstractproperty
    def fit_model(self):
        return self._fit_model

    @abstractproperty
    def fit_function(self):
        return self._fit_function

    @abstractproperty
    def scale_image(self):
        """
        Scale Image is a string or int value property that needs to be transmitted.
        """
        return self._scale_image

    @abstractmethod
    def get_pixel_args(
        self, img: np.ndarray, seg: np.ndarray, *args
    ) -> zip[tuple[tuple, np.ndarray]]:
        pass  # TODO: Chech weather the expected return type is correct

    @abstractmethod
    def get_seg_args(
        self,
        img: RadImgArray | np.ndarray,
        seg: SegImgArray,
        seg_number: int,
        *args,
    ) -> zip[tuple[list, np.ndarray]]:
        pass


class Parameters(Params):
    """Base class for all fitting parameters subclasses.

    Contains the basic attributes and methods for all DWI fitting parameters subclasses.

    Attributes:
        max_iter (int): Maximum number of iterations for fitting
        boundaries (Boundaries): Boundaries object containing fitting boundaries
        n_pools (int): Number of pools for fitting
        fit_area (str): Area of fitting  # TODO: Check if this is still needed
    """

    def __init__(self, json_file: str | Path | None = None):
        """Initializes basic Parameters object.

        Args:
            params_json (str | Path): Path to .json containing fitting parameters
        """
        # Set Basic Parameters
        self.json = Path()
        self.b_values = None
        self.max_iter = None
        if not hasattr(self, "boundaries") or self.boundaries is None:
            self.boundaries = Boundaries()
        self.n_pools = None
        self.fit_area = None
        self.fit_model = lambda: None
        self._fit_function = lambda: None
        self._scale_image: str | int = ""

        if isinstance(json_file, (str, Path)):
            self.json = json_file
            if self.json.is_file():
                self._load_json()
            else:
                print("Warning: Can't find parameter file!")
                self.json = Path()

    @property
    def json(self):
        """Path to .json file containing fitting parameters."""
        return self._json

    @json.setter
    def json(self, value: Path | str):
        if isinstance(value, str):
            value = Path(value)
        self._json = value

    @property
    def b_values(self):
        """b-values for fitting."""
        return self._b_values

    @b_values.setter
    def b_values(self, values: np.ndarray | list | None):
        if isinstance(values, list):
            values = np.array(values)
        elif values is None:
            self._b_values = np.array([])
        if isinstance(values, np.ndarray):
            self._b_values = np.expand_dims(values.squeeze(), axis=1)

    @property
    def fit_model(self):
        """Model function for fitting."""
        return self._fit_model

    @fit_model.setter
    def fit_model(self, method):
        if not isinstance(method, (Callable, partial)):
            raise ValueError("Fit Model must be a function or partial function.")
        self._fit_model = method

    @property
    def fit_function(self):
        """Fitting function for fitting."""
        return self._fit_function

    @fit_function.setter
    def fit_function(self, method):
        if not isinstance(method, (Callable, partial)):
            raise ValueError("Fit Function must be a function or partial function.")
        self._fit_function = method

    @property
    def scale_image(self):
        """Handles scaling of image for fitting."""
        return self._scale_image

    @scale_image.setter
    def scale_image(self, value):
        if not isinstance(value, (str, int)):
            raise ValueError("Scale Image must be a string or int value.")
        self._scale_image = value
        self.boundaries.scaling = value

    def load_from_json(self, json_file: str | Path):
        """Loads fitting parameters from .json file.

        Main method to import fitting parameters from .json file.

        Args:
            params_json (str | Path): Path to .json file containing fitting parameters
        """
        self.json = json_file
        self._load_json()

    def _load_json(self):
        with self.json.open("r") as file:
            params_dict = json.load(file)

        # Check if .json contains Class identifier and if .json and Params set match
        if "Class" not in params_dict.keys():
            # print("Error: Missing Class identifier!")
            # return
            raise ClassMismatch("Error: Missing Class identifier!")
        # elif not isinstance(self, globals()[params_dict["Class"]]):
        #         #     raise ClassMismatch("Error: Wrong parameter.json for parameter Class!")
        else:
            params_dict.pop("Class", None)
            for key, item in params_dict.items():
                # if isinstance(item, list):
                if hasattr(self, key):
                    if key == "boundaries":
                        self.boundaries.load(item)
                    else:
                        setattr(self, key, item)
                else:
                    print(
                        f"Warning: There is no {key} in the selected Parameter set!"
                        + f"{key} is skipped."
                    )

    def save_to_json(self, file_path: Path):
        """Saves fitting parameters to .json file.

        Args:
            file_path (Path): Path to .json file
        """
        attributes = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and not attr.startswith("_")
            and not isinstance(getattr(self, attr), partial)
        ]
        data_dict = dict()
        data_dict["Class"] = self.__class__.__name__
        for attr in attributes:
            # Custom Encoder

            if attr == "boundaries":
                value = getattr(self, attr).save()
            elif isinstance(getattr(self, attr), np.ndarray):
                value = getattr(self, attr).squeeze().tolist()
            elif isinstance(getattr(self, attr), Path):
                value = getattr(self, attr).__str__()
            else:
                value = getattr(self, attr)
            data_dict[attr] = value
        if not file_path.exists():
            with file_path.open("w") as file:
                file.write("")
        with file_path.open("w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        print(f"Parameters saved to {file_path}")

    def load_b_values(self, file: str | Path):
        """Loads b-values from file."""
        with open(file, "r") as f:
            self.b_values = np.array([int(x) for x in f.read().split("\n")])

    def get_pixel_args(
        self, img: np.ndarray | RadImgArray, seg: np.ndarray | SegImgArray, *args
    ) -> zip[tuple[tuple, np.ndarray]]:
        """Returns zip of tuples containing pixel arguments

        Basic method for packing pixel arguments for fitting. Enables multiprocessing.

        Args:
            img (np.ndarray): Image data
            seg (np.ndarray): Segmentation data
            *args: Additional arguments
        Returns:
            zip: Zip of tuples containing pixel arguments [(i, j, k), img[i, j, k, :]]
        """
        # zip of tuples containing a tuple and a nd.array
        pixel_args = zip(
            ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
            (img[i, j, k, :] for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))),
        )
        return pixel_args

    def get_seg_args(
        self, img: RadImgArray | np.ndarray, seg: SegImgArray, seg_number: int, *args
    ) -> zip[tuple[list, np.ndarray]]:
        """Returns zip of tuples containing segment arguments

        Similar to the get_pixel_args method, but for segment fitting.

        Args:
            img (np.ndarray): Image data
            seg (NiiSeg): Segmentation data
            seg_number (int): Segment number
            *args: Additional arguments
        Returns:
            zip: Zip of tuples containing segment arguments [(seg_number), mean_signal]
        """
        mean_signal = tools.get_mean_signal(img, seg, seg_number)
        return zip([[seg_number]], [mean_signal])
