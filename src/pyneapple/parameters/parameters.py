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
from .. import models


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
            raise ImportError(
                "tomlkit library is required for writing TOML files in Python 3.11+. "
                "Please install it with 'pip install tomlkit'"
            )
    else:
        try:
            import tomli_w

            tomli_w.dump(data, file_obj)
        except ImportError:
            raise ImportError(
                "tomli-w library is required for writing TOML files in Python < 3.11. "
                "Please install it with 'pip install tomli-w'"
            )


class AbstractParams(ABC):
    """Abstract base class for Parameters child class.

    Defines abstract properties and methods for Parameters child classes.
    """

    def __init__(self):
        self.file = Path()
        self.description: str | None = None
        self._fit_type = ""
        self._model: str | None = None
        if not hasattr(
            self, "_fit_model"
        ):  # Ensure _fit_model is defined but don't override if already set
            self._fit_model = lambda: None
        self._fit_function = lambda: None
        self.max_iter = None
        self.n_pools: int | None = None
        self.fit_tolerance: float = 1e-6
        self.b_values = None

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
        if not hasattr(self, "boundaries") or self.boundaries is None:
            self.boundaries = Boundaries()
        self.n_pools = None
        self._fit_function = None

        if isinstance(file, (str, Path)):
            self.file = file
            if self.file.is_file():
                # Choose loader based on file extension
                if self.file.suffix.lower() == ".toml":
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
            error_msg = (
                f"Unsupported fit_type: {value}. Must be 'single', 'multi', or 'gpu'."
            )
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
        self._set_parameters_from_dict(params_dict)

    def _load_toml(self):
        """Loads fitting parameters from .toml file."""
        try:
            with self.file.open("rb") as file:
                params_dict = tomllib.load(file)
            self._set_parameters_from_dict(params_dict)
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Failed to parse TOML file {self.file}: {e}")
            raise

    def _set_parameters_from_dict(self, params_dict: dict):
        if not "Class" in params_dict["General"]:
            warn_msg = (
                "Error: Class identifier not found in parameter file General Section!"
            )
            logger.error(warn_msg)
            raise ClassMismatch(warn_msg)
        else:
            general_params = params_dict["General"]
            logger.info(f"Loading parameters with class {general_params['Class']}")
            general_params.pop("Class", None)
            # load general parameters into the class attributes
            for key, item in general_params.items():
                try:
                    setattr(self, key, self._import_type_conversion(item))
                except AttributeError:
                    warn_msg = f"Parameter '{key}' not found in General Section!"
                    logger.warning(warn_msg)
            # load model parameters into model attributes
            if "Model" in params_dict:
                model_params = params_dict.get("Model")
                self._set_model_parameters(model_params)
            else:
                warn_msg = "No Model Section found in parameter file."
                logger.warning(warn_msg)
            # load boundaries if available
            try:
                for key in params_dict:
                    # legacy support for "boundaries" key
                    if isinstance(key, str) and key.lower() == "Boundaries".lower():
                        break
                self.boundaries.load(params_dict[key])
            except KeyError:
                warn_msg = f"Parameter 'Boundaries' not found in file!"
                logger.warning(warn_msg)

    def _set_model_parameters(self, model_params: dict):
        """Sets model parameters from a dictionary.

        Args:
            model_params (dict): Dictionary containing model parameters.
        """
        for key, item in model_params.items():
            try:
                if key in ["model", "name"]:
                    setattr(self.fit_model, "name", self._import_type_conversion(item))
                else:
                    setattr(self.fit_model, key, self._import_type_conversion(item))
            except AttributeError:
                warn_msg = f"Parameter '{key}' not found in Model Section!"
                logger.warning(warn_msg)

    def _prepare_data_for_saving(self) -> dict:
        attributes = self._get_attributes(self)
        data_dict = dict()
        data_dict["General"] = {}
        data_dict["General"]["Class"] = self.__class__.__name__

        for attr in attributes:
            # Custom Encoder
            if not attr in ["boundaries", "fit_model", "fit_function"]:
                # Skip attributes that are not to be saved
                value = getattr(self, attr)
                value = self._export_type_conversion(value)
                data_dict["General"][attr] = value
            elif attr.lower() == "boundaries":
                value = getattr(self, attr.lower()).save()
                data_dict["Boundaries"] = value
            elif attr in ["fit_model"]:
                for key in self._get_attributes(getattr(self, attr)):
                    if not key in ["model", "args"]:
                        value = getattr(getattr(self, attr), key)
                        if key == "name":
                            key = "model"
                        value = self._export_type_conversion(value)
                        if not "Model" in data_dict:
                            data_dict["Model"] = {}
                        data_dict["Model"][key] = value
            else:
                continue

        return data_dict

    @staticmethod
    def _get_attributes(obj):
        return [
            attr
            for attr in dir(obj)
            if not callable(getattr(obj, attr))
            and not attr.startswith("_")
            and not isinstance(getattr(obj, attr), partial)
        ]

    @staticmethod
    def _export_type_conversion(value):
        # Datatype conversion
        if isinstance(value, np.ndarray):
            value = value.squeeze().tolist()
        elif isinstance(value, Path):
            value = value.__str__()
        elif value is None:
            value = ""
        return value

    @staticmethod
    def _import_type_conversion(value):
        if isinstance(value, str):
            if not value:
                value = None
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif Path(value).exists():
                value = Path(value)
        return value

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
            (
                (int(i), int(j), int(k))
                for i, j, k in zip(*np.nonzero(np.squeeze(seg, axis=3)))
            ),
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
