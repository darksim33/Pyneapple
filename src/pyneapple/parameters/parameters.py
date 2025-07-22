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

Version History:
    1.3.3 (2024-09-18): Refactored Parameters class to BaseParams class.
    1.5.0 (2024-12-06): Reworked Parameter classes to better integrate with gpu fitting.
    1.6.1 (2024-MM-DD): Added TOML support for parameter loading.
"""

from __future__ import annotations
from collections.abc import Callable
import sys
import json
from functools import partial
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np

# Import appropriate TOML library based on Python version
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from ..utils.logger import logger
from radimgarray import RadImgArray, SegImgArray, tools
from ..utils.exceptions import ClassMismatch
from ..parameters import Boundaries


def toml_dump(data, file_obj):
    """
    Dump data as TOML to a file object, using the appropriate library based on Python version.

    Args:
        data (dict): The data to serialize as TOML
        file_obj: A file-like object opened in the appropriate mode

    Raises:
        ImportError: If the required TOML writing library is not available
    """
    if sys.version_info >= (3, 11):
        try:
            import tomlkit
            file_obj.write(tomlkit.dumps(data))
        except ImportError:
            raise ImportError("tomlkit library is required for writing TOML files in Python 3.11+. "
                              "Please install it with 'pip install tomlkit'")
    else:
        try:
            import tomli_w
            tomli_w.dump(data, file_obj)
        except ImportError:
            raise ImportError("tomli-w library is required for writing TOML files in Python < 3.11. "
                              "Please install it with 'pip install tomli-w'")


class AbstractParams(ABC):
    """Abstract base class for Parameters child class.

    Defines abstract properties and methods for Parameters child classes.
    """

    def __init__(self):
        self._model: str = ""
        self._fit_type = ""
        self._fit_model = lambda: None
        self._fit_function = lambda: None
        self._comment: str = ""
        self.fit_reduced: bool = False
        self.fit_tolerance: float = 1e-6

    @property
    @abstractmethod
    def model(self):
        return self._model

    @property
    @abstractmethod
    def fit_type(self):
        return self._fit_type

    @property
    @abstractmethod
    def fit_model(self):
        return self._fit_model

    @property
    @abstractmethod
    def fit_function(self):
        return self._fit_function

    # @abstractmethod
    # @property
    # def scale_image(self):
    #     """
    #     Scale Image is a string or int value property that needs to be transmitted.
    #     """
    #     return self._scale_image

    @abstractmethod
    def get_pixel_args(
            self, img: np.ndarray, seg: np.ndarray, *args
    ) -> zip[tuple[tuple, np.ndarray]]:
        pass  # TODO: Check weather the expected return type is correct

    @abstractmethod
    def get_seg_args(
            self,
            img: RadImgArray | np.ndarray,
            seg: SegImgArray,
            seg_number: int,
            *args,
    ) -> zip[tuple[list, np.ndarray]]:
        pass


class BaseParams(AbstractParams):
    """Base class for all fitting parameters subclasses.

    Contains the basic attributes and methods for all DWI fitting parameters subclasses.

    Attributes:
        model (str): Model name for fitting.
        fit_type (str): Type of fitting.
        fit_model (function): Model function for fitting.
        fit_function (function): Fitting function for fitting.
        scale_image (str | int): Scale Image property for fitting.
        fit_reduced (bool): Flag for fit_reduced fitting.
        fit_tolerance (float): Tolerance for gpu based fitting.
        max_iter (int): Maximum number of iterations for fitting
        boundaries (Boundaries): Boundaries object containing fitting boundaries
        n_pools (int): Number of pools for fitting
    """

    def __init__(self, file: str | Path | None = None):
        """Initializes basic Parameters object.

        Args:
            params_json (str | Path): Path to .json containing fitting parameters
        """
        super().__init__()
        # Set Basic Parameters
        self.file = Path()
        self.b_values = None
        self.max_iter = None
        if not hasattr(self, "boundaries") or self.boundaries is None:
            self.boundaries = Boundaries()
        self.n_pools = None
        self._fit_model = None
        self._fit_function = None
        self._scale_image: str | int = ""

        if isinstance(file, (str, Path)):
            self.file = file
            if self.file.is_file():
                # Choose loader based on file extension
                if self.file.suffix.lower() == '.toml':
                    self._load_toml()
                else:
                    self._load_json()
            else:
                logger.warning(f"Can't find parameter file {file}!")
                self.file = Path()

    @property
    def model(self):
        """Model name for fitting."""
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    @property
    def fit_type(self):
        """Type of fitting."""
        return self._fit_type

    @fit_type.setter
    def fit_type(self, value: str):
        if value.lower() not in ("single", "multi", "gpu"):
            error_msg = f"Unsupported fit_type: {value}. Must be 'single', 'multi', or 'gpu'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Setting fit_type to {value}")
        self._fit_type = value

    @property
    def fit_model(self):
        """Model function for fitting."""
        return self._fit_model

    @fit_model.setter
    def fit_model(self, model):
        self._fit_model = model

    @property
    def fit_function(self):
        """Fitting function for fitting."""
        return self._fit_function

    @fit_function.setter
    def fit_function(self, method):
        if not isinstance(method, (Callable, partial)):
            error_msg = "Fit Function must be a function or partial function."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self._fit_function = method

    @property
    def file(self):
        """Path to .json file containing fitting parameters."""
        return self._file

    @file.setter
    def file(self, value: Path | str):
        if isinstance(value, str):
            value = Path(value)
        self._file = value

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

    # @property
    # def scale_image(self):
    #     """Handles scaling of image for fitting."""
    #     return self._scale_image
    #
    # @scale_image.setter
    # def scale_image(self, value):
    #     if not isinstance(value, (str, int)):
    #         raise ValueError("Scale Image must be a string or int value.")
    #     self._scale_image = value
    #     self.boundaries.scaling = value

    def load_from_json(self, json_file: str | Path):
        """Loads fitting parameters from .json file.

        Main method to import fitting parameters from .json file.

        Args:
            json_file(str | Path): Path to .json file containing fitting parameters
        """
        self.file = json_file
        self._load_json()

    def load_from_toml(self, toml_file: str | Path):
        """Loads fitting parameters from .toml file.

        Main method to import fitting parameters from .toml file.

        Args:
            toml_file(str | Path): Path to .toml file containing fitting parameters
        """
        self.file = toml_file
        self._load_toml()

    def _load_json(self):
        with self.file.open("r") as file:
            params_dict = json.load(file)

        # Check if .json contains Class identifier and if .json and Params set match
        if "Class" not in params_dict.keys():
            raise ClassMismatch("Error: Missing Class identifier!")
        # elif not isinstance(self, globals()[params_dict["Class"]]):
        #         #     raise ClassMismatch("Error: Wrong parameter.json for parameter Class!")
        else:
            logger.debug(f"Loading parameters with class {params_dict['Class']}")
            params_dict.pop("Class", None)
            for key, item in params_dict.items():
                # if isinstance(item, list):
                if hasattr(self, key):
                    if key == "boundaries":
                        self.boundaries.load(item)
                    else:
                        setattr(self, key, item)
                else:
                    logger.warning(f"Parameter '{key}' not found in the selected Parameter set. Skipping.")

    def _load_toml(self):
        """Loads fitting parameters from .toml file."""
        try:
            with self.file.open("rb") as file:
                params_dict = tomllib.load(file)

            # Check if .toml contains Class identifier and if .toml and Params set match
            if "Class" not in params_dict.keys():
                raise ClassMismatch("Error: Missing Class identifier!")
            else:
                logger.debug(f"Loading parameters with class {params_dict['Class']}")
                params_dict.pop("Class", None)
                for key, item in params_dict.items():
                    if hasattr(self, key):
                        if key == "boundaries":
                            self.boundaries.load(item)
                        else:
                            setattr(self, key, item)
                    else:
                        logger.warning(f"Parameter '{key}' not found in the selected Parameter set. Skipping.")
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Failed to parse TOML file {self.file}: {e}")
            raise

    def _prepare_data_for_saving(self) -> dict:
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
            elif attr in ["fit_model", "fit_function"]:
                continue
            elif isinstance(getattr(self, attr), np.ndarray):
                value = getattr(self, attr).squeeze().tolist()
            elif isinstance(getattr(self, attr), Path):
                value = getattr(self, attr).__str__()
            else:
                value = getattr(self, attr)
            data_dict[attr] = value
        return data_dict

    def save_to_json(self, file_path: Path):
        """Saves fitting parameters to .json file.

        Args:
            file_path (Path): Path to .json file
        """
        data_dict = self._prepare_data_for_saving()
        logger.debug(f"Saving {self.__class__.__name__} parameters to {file_path}")

        if not file_path.exists():
            with file_path.open("w") as file:
                file.write("")
        with file_path.open("w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        logger.info(f"Parameters saved to {file_path}")

    def save_to_toml(self, file_path: Path):
        """Saves fitting parameters to .toml file.

        Note: This method requires a TOML writer library like 'tomli-w' which is
        not included by default. Users need to install it separately if they want
        to use this functionality.

        Args:
            file_path (Path): Path to .toml file
        """
        data_dict = self._prepare_data_for_saving()
        logger.debug(f"Saving {self.__class__.__name__} parameters to {file_path}")

        if not file_path.exists():
            with file_path.open("w") as file:
                file.write("")

        try:
            # For Python >= 3.11, tomlkit accepts a text file
            # For Python < 3.11, tomli_w requires a binary file
            mode = "w" if sys.version_info >= (3, 11) else "wb"
            with file_path.open(mode) as toml_file:
                toml_dump(data_dict, toml_file)
            logger.info(f"Parameters saved to {file_path}")
        except ImportError as e:
            logger.error(f"{e}")
            raise

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
            img (np.ndarray): Image data (4D)
            seg (np.ndarray): Segmentation data (4D [x,y,z,1])
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
