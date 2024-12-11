"""Module for handling boundaries for fitting parameters."""

from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class BoundariesBase(ABC):
    """Basic abstract boundaries class"""

    def __init__(self):
        self._scaling: str | int | float | list | None = None
        self._parameter_names = None
        self.dict = dict()

    @abstractmethod
    def load(self, _dict: dict):
        """Load dict into class."""
        pass

    @abstractmethod
    def save(self) -> dict:
        """Return dict for saving to json"""
        pass

    @abstractproperty
    def parameter_names(self) -> list | dict | None:
        """Get list of parameter names."""
        return self._parameter_names

    @abstractproperty
    def scaling(self):
        """Scaling to parameters id needed."""
        return self._scaling

    @abstractmethod
    def apply_scaling(self, value: list) -> list:
        """Apply scaling to parameter values."""
        pass

    @abstractmethod
    def get_axis_limits(self) -> tuple:
        """Get Limits for axis in parameter values."""
        pass


class Boundaries(BoundariesBase):
    """Basic boundaries class for IVIM and NNLS

    Attributes:
        values (dict): Dictionary for storing values
        scaling (str | int | float | list | None): Scaling factor or string
        dict (dict): Dictionary for storing values
        number_points (int): Number of points for creating spectral array element

    Methods:
        load: Load dict into class
        save: Return dict for saving to json
        apply_scaling: Apply scaling to parameter values
        get_axis_limits: Get Limits for axis in parameter values
        get_boundary_names: Get names of all boundaries as a list
    """

    def __init__(self):
        """Initiation for basic boundaries class for IVIM and NNLS"""
        self.values = dict()
        self._scaling: str | int | float | list | None = None
        self._parameter_names = None
        # a factor or string (needs to be added to apply_scaling to boundaries)
        self.dict = dict()
        self.number_points = 250  # reserved for creating spectral array element. behaves like a resolution

    def load(self, _dict: dict):
        """Load dict into class.

        Args:
            _dict (dict): Dictionary to be loaded
        """
        self.dict = _dict.copy()

    def save(self) -> dict:
        """Return dict for saving to json"""
        _dict = self.dict.copy()
        for key, value in _dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        value[sub_key] = sub_value.tolist()
            if isinstance(value, np.ndarray):
                _dict[key] = value.tolist()
        return _dict

    @property
    def parameter_names(self) -> list | dict | None:
        """Returns parameter names from json for IVIM (and generic names vor NNLS)"""
        return None

    @parameter_names.setter
    def parameter_names(self, data: dict | list | None):
        self._parameter_names = data

    @property
    def scaling(self):
        """Scaling to parameters if needed."""
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    def apply_scaling(self, value: list) -> list:
        """Apply scaling to parameter values."""
        return value

    def get_axis_limits(self) -> tuple:
        """Get Limits for axis in parameter values."""
        return 0.0001, 1

    def get_boundary_names(self) -> list:
        """Return names of all boundaries as a list."""
        names = []
        for key, value in self.dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    names.append(key + "_" + sub_key)
            else:
                names.append(key)
        return names


class IVIMBoundaries(Boundaries):
    """Handle IVIM fitting boundaries.

    Attributes:
        dict (dict): Dictionary for storing values
        scaling (str | int | float | list | None): Scaling factor or string
        number_points (int): Number of points for creating spectral array element
    Methods:
        load: Load dict into class
        save: Return dict for saving to json
        apply_scaling: Apply scaling to parameter values
        get_axis_limits: Get Limits for axis in parameter values
    """

    def __init__(self):
        """
        Handle IVIM fitting boundaries.

        Boundaries imported by loading a dict. The dict should have the following
        structure:
            "D":  {<compartment>: [x0, lb, ub], ... } the compartments should increase
                from slowest to fastest.
            "f": { <compartment>: [x0, lb, ub], ... } the compartments should increase
                from slowest to fastest.
            "S": { "0": [x0, lb, ub] }  # S0 is always the last parameter

        Optional:
        "T1": [x0, lb, ub]  # T1 is optional and can be added to the dict.

        Data is imported using the load() method.
        """
        self.dict: dict = dict()
        self.parameter_names = None
        super().__init__()

    @property
    def parameter_names(self) -> list | dict | None:
        """Returns parameter names from json for IVIM (and generic names vor NNLS)

        Returns:
            names (list | None): List of parameter names
        """
        return self._parameter_names

    @parameter_names.setter
    def parameter_names(self, data: dict | list | None):
        if isinstance(data, (dict | list)):
            self._parameter_names = data
        else:
            names = list()
            for key in self.dict:
                for subkey in self.dict[key]:
                    names.append(key + "_" + subkey)
                    names = self.apply_scaling(names)
            if len(names) == 0:
                names = None
            self._parameter_names = names

    @property
    def scaling(self):
        """Scaling to parameters if needed."""
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    @property
    def start_values(self):
        """Get start values for IVIM parameters."""
        return self._get_boundary(0)

    @start_values.setter
    def start_values(self, x0: list | np.ndarray):
        self._set_boundary(0, x0)

    @property
    def lower_bounds(self):
        """Get lower stop values for IVIM parameters."""
        return self._get_boundary(1)

    @lower_bounds.setter
    def lower_bounds(self, lb: list | np.ndarray):
        self._set_boundary(1, lb)

    @property
    def upper_bounds(self):
        """Get upper stop values for IVIM parameters."""
        return self._get_boundary(2)

    @upper_bounds.setter
    def upper_bounds(self, ub: list | np.ndarray):
        self._set_boundary(2, ub)

    def load(self, data: dict):
        """Load dict into class.
        Data Shape:
        "D": {
            "<NAME>": [x0, lb, ub],
            ...
        },
        "f": {
            "<NAME>": [x0, lb, ub],
            ...
        }
        "S": {
            "<NAME>": [x0, lb, ub],
        }
        """
        self.dict = data.copy()

    def save(self) -> dict:
        """Return dict for saving to json"""
        _dict = super().save()
        return _dict

    def apply_scaling(self, value: list) -> list:
        """Apply scaling to parameter values.

        Args:
            value (list): List of parameter values
        """
        if isinstance(self._scaling, str):
            if self.scaling == "S/S0" and "S" in self.dict.keys():
                # with S/S0 the number of Parameters is reduced.
                value = value[:-1]
        elif isinstance(self.scaling, (int, float)):
            pass
        return value

    def _get_boundary(self, pos: int) -> np.ndarray:
        """Get boundary values for IVIM parameters.

        Shape of boundary values:
            [f1,D1,f2,D2,...,fn,Dn(,TM)] or
            [f1,D1,f2,D2,...,Dn-1,Dn(,TM)] for reduced fitting.
        """
        d_values, fractions, additional = list(), list(), list()
        for key in self.dict:
            if key == "D":
                for subkey in self.dict[key]:
                    d_values.append(self.dict[key][subkey][pos])
            elif key == "f":
                for subkey in self.dict[key]:
                    fractions.append(self.dict[key][subkey][pos])
            else:
                for subkey in self.dict[key]:
                    additional.append(self.dict[key][subkey][pos])

        if len(fractions) == len(d_values):
            values = [item for pair in zip(fractions, d_values) for item in pair]
        elif len(fractions) == len(d_values) - 1:  # reduced fitting
            values = [item for pair in zip(fractions, d_values[:-1]) for item in pair]
            values.append(d_values[-1])
        elif len(fractions) == len(d_values) + 1:  # segmented fitting
            values = [item for pair in zip(fractions[:-1], d_values) for item in pair]
            values.append(fractions[-1])
        else:
            raise ValueError(
                "Length of fractions and D values do not match (n==n or n==n+1)."
            )
        if len(additional) > 0:
            values = values + additional

        return np.array(values)

        # for key in self.dict:
        #     for subkey in self.dict[key]:
        #         values.append(self.dict[key][subkey][pos])
        # values = self.apply_scaling(values)
        # values = np.array(values)
        # return np.array(values)

    def _set_boundary(self, pos: int, values: list | np.ndarray):
        idx = 0
        for key in self.dict:
            for subkey in self.dict[key]:
                self.dict[key][subkey][pos] = values[idx]
                idx += 1

    def get_axis_limits(self) -> tuple:
        """Get Limits for plot axis from parameter values."""
        _min = min(self.lower_bounds)  # this should always be the lowest D value
        d_values = list()
        for key in self.dict["D"]:
            d_values = d_values + self.dict["D"][key]
        _max = max(d_values)

        # _max = max(self.upper_stop_values)
        return _min, _max


class NNLSBoundaries(Boundaries):
    """Handle NNLS fitting boundaries.

    Boundaries imported by loading a dict. The dict should have the following structure:
    "boundaries": {
        "d_range": [],
        "n_bins": []
    }
    Parameters are read starting with the first key descending to bottom level
    followed by the next key.

    Attributes:
        dict (dict): Dictionary for storing values
        scaling (str | int | float | list | None): Scaling factor or string
        number_points (int): Number of points for creating spectral array element

    Methods:
        load: Load dict into class
        save: Return dict for saving to json
        apply_scaling: Apply scaling to parameter values
        get_axis_limits: Get Limits for axis in parameter values
    """

    def __init__(self):
        self._scaling = None
        self.dict = dict()
        self.parameter_names = None
        super().__init__()

    def load(self, data: dict):
        """
        The dictionaries need to be shape according to the following shape:
        "boundaries": {
            "d_range": [],
            "n_bins": []
        }
        Parameters are read starting with the first key descending to bottom level
        followed by the next key.
        """
        self.dict = data
        self.number_points = data["n_bins"]

    @property
    def parameter_names(self) -> list | dict | None:
        """Returns parameter names for NNLS"""
        return self._parameter_names

    @parameter_names.setter
    def parameter_names(self, data: dict | list | None):
        if data is not None:
            self._parameter_names = data
        else:
            self._parameter_names = [f"X{i}" for i in range(0, 10)]

    @property
    def scaling(self):
        """Scaling to parameters if needed."""
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    def apply_scaling(self, value: list) -> list:
        """Currently there is no scaling available for NNLS (except CV)."""
        return value

    def get_axis_limits(self) -> tuple:
        return self.dict.get("d_range", [0])[0], self.dict.get("d_range", [0, 0])[1]
